"""
app.py
======
Gradio 前端：与 FastAPI 后端交互，展示 Agent 流式输出。

功能：
  - 多轮对话（Gradio 原生管理历史）
  - 流式打字机效果
  - 额外面板显示 sources 和 metrics

启动：
    python src/app.py
    （需先启动后端：uvicorn src.api.main:app --port 8000）
"""

import json

import gradio as gr
import requests

API_URL = "http://localhost:8000/query"


# ---------------------------------------------------------------------------
# SSE 流式请求
# ---------------------------------------------------------------------------

def stream_query(query: str, history: list[list[str]]):
    """
    向 FastAPI 发送 SSE 请求，流式 yield (partial_answer, sources, metrics)。

    Args:
        query   : 用户当前问题
        history : Gradio 格式历史 [[user, assistant], ...]

    Yields:
        tuple: (partial_answer: str, sources: str, metrics: str)
    """
    # 将 Gradio history 转成 API 格式
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    payload = {"query": query, "messages": messages}
    answer  = ""
    sources = ""
    metrics = ""

    with requests.post(API_URL, json=payload, stream=True, timeout=120) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data:"):
                continue
            try:
                event = json.loads(decoded[len("data:"):].strip())
            except json.JSONDecodeError:
                continue

            etype = event.get("event")

            if etype == "token":
                answer += event.get("text", "")
                yield answer, sources, metrics

            elif etype == "retrieved":
                yield answer, sources, f"Retrieved {event.get('count', 0)} chunks..."

            elif etype == "reflection":
                yield answer, sources, f"Reflection: {event.get('text', '')}"

            elif etype == "done":
                src_list = event.get("sources", [])
                m        = event.get("metrics", {})
                sources  = "\n".join(f"- `{s}`" for s in src_list)
                metrics  = (
                    f"**Latency:** {m.get('latency_ms', 0)} ms  \n"
                    f"**Prompt tokens:** {m.get('prompt_tokens', 0)}  \n"
                    f"**Completion tokens:** {m.get('completion_tokens', 0)}  \n"
                    f"**Retries:** {m.get('retry_count', 0)}"
                )
                yield answer, sources, metrics

            elif etype == "error":
                yield f"⚠️ Error: {event.get('message', '')}", sources, metrics


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Structured RAG — FinQA Assistant") as demo:
    gr.Markdown("# 📊 Structured RAG — FinQA Assistant")

    with gr.Row():
        # 左栏：聊天
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, label="Chat")
            msg_box = gr.Textbox(
                placeholder="Ask about FinQA documents...",
                label="Your question",
                show_label=False,
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn  = gr.Button("Clear")

        # 右栏：Sources + Metrics
        with gr.Column(scale=1):
            sources_box = gr.Markdown(label="Sources", value="*No sources yet.*")
            metrics_box = gr.Markdown(label="Metrics", value="*No metrics yet.*")

    # ---------------------------------------------------------------------------
    # 事件处理
    # ---------------------------------------------------------------------------

    def respond(message, history):
        """流式更新 chatbot + sources + metrics。"""
        history = history or []
        partial_answer = ""

        for answer, sources, metrics in stream_query(message, history):
            partial_answer = answer
            # 将当前 partial answer 附加到历史末尾（流式更新最后一条）
            updated_history = history + [[message, partial_answer]]
            yield updated_history, "", sources, metrics

    submit_btn.click(
        fn=respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box, sources_box, metrics_box],
    )
    msg_box.submit(
        fn=respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box, sources_box, metrics_box],
    )
    clear_btn.click(
        fn=lambda: ([], "", "*No sources yet.*", "*No metrics yet.*"),
        outputs=[chatbot, msg_box, sources_box, metrics_box],
    )


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
