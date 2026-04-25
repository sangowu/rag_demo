# RAG 项目改造 TODO

## 进行中

- [ ] **Ablation Study 执行（需重建完整索引）**
  - `scripts/ablation_study.py` 已写好（5 配置，n=200，seed=42）
  - EC2 上已跑通，但 chunks_baseline 仅 607 条（~200 docs），eval 从全量 7134 条随机采样
  - 问题：200 条 query 的 gold_doc 大部分不在 607 条索引里，导致 Hit@5 仅 6%（非真实性能）
  - 修复方案：重建完整索引（全量 2408 docs）或过滤 eval 只保留已索引 doc_id 的 query
  - 产出：`data/results/ablation_results.json` + `ablation_chart.png`

- [ ] **LegalBench-RAG-mini 接入**（跨域泛化验证）
  - 72 legal docs / 776 queries，CC-BY 开源
  - 需新增：`scripts/ingest_legalbench.py`、`scripts/convert_legalbench.py`
  - ChunkManager 需增加 char_start/char_end 偏移字段
  - eval_smoke.py 需支持 char-level GT 匹配

## AWS 云部署（最高优先级，JD 频率 40.1%）

- [x] **S3 Bucket 创建**
  - Bucket 名称：`rag-documents-sango`（eu-west-1）
  - BM25 索引自动上传：`bm25/bm25_*.pkl`
  - `src/bm25_store.py` 已改造：build 时上传 S3，load 时从 S3 下载

- [x] **RDS PostgreSQL**
  - 实例：`rag-database-1`（db.t3.micro，eu-west-1）
  - pgvector 0.8.1 已开启
  - 数据库：`rag_demo`
  - 连接串通过 Secrets Manager 注入（不写在代码里）

- [x] **Secrets Manager**
  - `rag-demo/pg-dsn`：PostgreSQL 完整连接串
  - `rag-demo/google-api-key`：Gemini API Key
  - ECS 容器启动时自动注入为环境变量

- [x] **ECR 镜像仓库**
  - 仓库：`rag-demo`
  - 镜像地址：`569260897196.dkr.ecr.eu-west-1.amazonaws.com/rag-demo:latest`

- [x] **ECS Fargate 容器部署**
  - 集群：`rag-demo-cluster`
  - 服务：`rag-demo-service`（1 Task，1vCPU/2GB）
  - Task Definition：`rag-demo:2`
  - FastAPI `/health` 返回正常

- [x] **IAM 权限配置**
  - `rag-ecs-execution-role`：拉取 ECR 镜像、读取 Secrets Manager、写 CloudWatch
  - `rag-ecs-task-role`：访问 S3、Secrets Manager

- [x] **CloudWatch 日志**
  - 日志组：`/ecs/rag-demo`
  - 收集容器 stdout/stderr

- [x] **GitHub Actions CI/CD**
  - 文件：`.github/workflows/deploy.yml`
  - 触发：push 到 main 分支
  - 流程：checkout → AWS 登录 → ECR 登录 → docker build/push → ECS 滚动更新

- [ ] **ALB（Application Load Balancer）**
  - 提供固定域名入口（当前每次重启 IP 变化）
  - 目标组指向 ECS Service
  - 完成标志：`http://<ALB-DNS>/health` 可访问

- [x] **EC2 g4dn.xlarge**
  - Tesla T4 / CUDA 13.0 / Ubuntu 24.04
  - BGE-M3 embedding server + BGE-reranker 已部署（port 8000）
  - systemd 开机自启（`/etc/systemd/system/embedding-server.service`）
  - 24小时无请求自动关机（IDLE_TIMEOUT=86400）
  - ECS 通过 EMBEDDING_SERVER_URL 环境变量连接（私有 IP 172.31.30.33）
  - 本地通过 `scripts/run_ec2.ps1` 一键唤醒 + 运行脚本 + 自动关机

- [ ] **ElastiCache Redis**
  - 替换 in-memory SemanticCache
  - 驱逐策略：allkeys-lfu
  - key 带模型版本号防止缓存污染
  - TTL 随机抖动防止缓存雪崩

- [ ] **Lambda + SQS 异步 Ingestion Pipeline**（可选）
  - S3 上传文档 → Lambda → SQS → Worker 消费
  - 解耦上传和处理，支持大批量并发摄入

- [ ] **自动化 Pipeline**（可选）
  - EventBridge 定时触发
  - 检测 S3 新文件 → 自动摄入 → 多 Agent 分析 → 报告存档

## 替换现有组件

- [x] **pgvector 替换 ChromaDB**
  - 迁移 `src/vector_store.py`
  - HNSW 索引参数：m=16, ef_construction=64，ef_search=100
  - GIN 索引支持 metadata 过滤

- [ ] **Redis 替换 in-memory SemanticCache**
  - 直接使用 ElastiCache（见 AWS 部分）

- [ ] **自建 async loop 替换 LangGraph**
  - 重写 `src/agent/` 目录
  - 原生 tool-use 循环（~30行）
  - PostgreSQL 持久化 session 状态

## 全新实现

- [ ] **MCP Server**
  - 新建 `src/mcp_server.py`
  - 依赖：fastmcp
  - 工具：search_knowledge_base / calculate / ingest_document
  - 硬超时保护（10s），返回结构化文本

- [x] **Late Chunking**
  - BGE-M3 输出 token 级向量，按 chunk 边界池化
  - 每个 chunk embedding 包含全文上下文

- [ ] **长上下文 Embedding 模型**
  - 评估 GTE-Qwen2-7B（32k token）
  - 对比 BGE-M3 在 FinQA 文档上的召回质量

## 升级现有组件

- [ ] **Summary Embedding**
  - 改为 embed 摘要向量而非原文向量
  - 检索用摘要向量，传入 LLM 用完整原文
  - 分级策略：<1000 token 全文，1000-4000 summary，>4000 Late Chunking

## 可观测性（后置）

- [ ] **Langfuse**
  - 替换或补充 LangSmith，eval 追踪 + 可视化面板

- [ ] **OpenTelemetry + Prometheus + Grafana**
  - 系统级监控，AWS 部署后可与 CloudWatch 并行

## 已完成

- [x] RRF 替换 alpha 加权融合（`src/retriever.py`）
- [x] `scripts/ablation_study.py` — 消融实验脚本，5 配置
- [x] `scripts/generate_ablation_chart.py` — 消融结果可视化
- [x] S3 Bucket 创建（`rag-documents-sango`）
- [x] RDS PostgreSQL + pgvector（`rag-database-1`）
- [x] Secrets Manager（pg-dsn + google-api-key）
- [x] ECR 镜像仓库（`rag-demo`）
- [x] ECS Fargate 服务（`rag-demo-service`，运行中）
- [x] CloudWatch 日志（`/ecs/rag-demo`）
- [x] GitHub Actions CI/CD（push 自动部署）
- [x] BM25 S3 集成（build 上传，load 下载）
