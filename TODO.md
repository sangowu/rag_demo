# RAG 项目改造 TODO

## 进行中
- [ ] （当前无）

## 替换现有组件

- [ ] **pgvector 替换 ChromaDB**
  - 迁移 `src/vector_store.py`
  - 建表：chunks（embedding + metadata JSONB）
  - HNSW 索引参数：m=16, ef_construction=64
  - 查询时设置 ef_search=100
  - 建 GIN 索引支持 metadata 过滤
  - VACUUM ANALYZE 写入批量数据后执行

- [ ] **Redis 替换 in-memory SemanticCache**
  - 迁移 `src/semantic_cache.py`
  - key 格式带模型版本号（防止模型升级后旧缓存污染）
  - 互斥锁防止缓存击穿
  - TTL 加随机抖动防止缓存雪崩
  - 驱逐策略：allkeys-lfu
  - 文档更新时标记脏数据

- [ ] **自建 async loop 替换 LangGraph**
  - 重写 `src/agent/` 目录
  - 原生 tool-use 循环（~30行）
  - PostgreSQL 持久化 session 状态

## 全新实现

- [ ] **MCP Server**
  - 新建 `src/mcp_server.py`
  - 依赖：fastmcp
  - 工具：search_knowledge_base / calculate / ingest_document
  - tool description 质量是关键，决定 Claude 调用行为
  - 硬超时保护（10s）
  - 返回结构化文本，不返回原始 JSON
  - 配置 `.claude/settings.json` 接入 Claude Code

- [ ] **Late Chunking**
  - 改造 ingestion pipeline
  - BGE-M3 输出 token 级向量，按 chunk 边界池化
  - 每个 chunk embedding 包含全文上下文
  - 无需额外 LLM 调用

- [ ] **长上下文 Embedding 模型**
  - 评估 GTE-Qwen2-7B（32k token）
  - 对比 BGE-M3 在 FinQA 文档上的召回质量
  - 中等长度文档（<4k token）可直接全文 embedding

## 升级现有组件

- [ ] **Summary Embedding**
  - `src/summary_store.py` 已有摘要生成基础
  - 改为 embed 摘要向量而非原文向量
  - 检索用摘要向量，传入 LLM 用完整原文
  - 配合文档长度分级策略：
    - < 1000 token：直接全文 embedding
    - 1000-4000 token：summary embedding
    - > 4000 token：Late Chunking

## 可观测性（后置）

- [ ] **Langfuse**
  - 替换或补充 LangSmith
  - eval 追踪 + 可视化面板

- [ ] **OpenTelemetry + Prometheus + Grafana**
  - 系统级监控：latency / token 用量 / 检索命中率
  - Docker Compose 加入 Prometheus + Grafana 服务

## 已完成

- [x] RRF 替换 alpha 加权融合（`src/retriever.py`）
