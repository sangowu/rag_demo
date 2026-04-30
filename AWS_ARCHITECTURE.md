# AWS 部署架构说明

## 整体架构图

```
用户请求
    ↓
ECS Fargate（FastAPI 主服务）
    ↓              ↓              ↓
RDS PostgreSQL   S3 Bucket    EC2 g4dn（待配额）
  pgvector       BM25 索引    embedding server
  向量存储        文档存储      + reranker
    ↓
Gemini API（外部 LLM）
    ↓
返回答案
```

---

## 每个服务的作用

### 1. IAM（Identity and Access Management）
**是什么：** AWS 的权限管理系统，控制"谁能做什么"。

**我们创建了什么：**
- `rag-developer`（IAM User）：你自己使用，拥有 CLI 和 GitHub Actions 的操作权限
- `rag-ecs-execution-role`（IAM Role）：ECS 启动容器时使用，用于拉取 ECR 镜像、读取 Secrets Manager、写 CloudWatch 日志
- `rag-ecs-task-role`（IAM Role）：容器运行时使用，用于访问 S3、Secrets Manager

**核心概念：**
- User = 人用的账号，有 Access Key
- Role = 服务用的账号，由 AWS 服务临时"扮演"
- Policy = 具体权限规则，附加到 User 或 Role 上

---

### 2. S3（Simple Storage Service）
**是什么：** 对象存储，类似网盘，存任意文件。

**Bucket 名称：** `rag-documents-sango`

**存什么：**
- `bm25/` 目录：BM25 稀疏索引（`.pkl` 文件）
- 未来：原始文档（PDF/Markdown）

**工作流程：**
```
本地 build_index.py 运行
    ↓
生成 BM25 索引（.pkl 文件）
    ↓
自动上传到 S3（bm25/bm25_baseline.pkl 等）
    ↓
ECS 容器启动时从 S3 下载索引到容器内存
```

**为什么用 S3 而不是直接打包进镜像：**
- BM25 索引是数据，镜像是代码，两者应该分离
- 索引更新不需要重新构建镜像
- 容器重启后数据不丢失

---

### 3. RDS（Relational Database Service）
**是什么：** AWS 托管的关系型数据库，我们用的是 PostgreSQL。

**实例名：** `rag-database-1`
**连接地址：** `rag-database-1.cn4eui80g8r4.eu-west-1.rds.amazonaws.com`
**数据库名：** `rag_demo`

**存什么：**
- `chunks` 表：文档切片 + BGE-M3 向量（pgvector 格式）
- HNSW 索引：加速向量相似度搜索

**为什么用 RDS 而不是本地 PostgreSQL：**
- 托管服务：自动备份、故障恢复
- 网络隔离：在 VPC 内部，不直接暴露公网
- ECS 容器可以通过内网连接，低延迟

**pgvector：** PostgreSQL 的向量扩展，支持 ANN（近似最近邻）搜索，我们用 HNSW 算法。

---

### 4. Secrets Manager
**是什么：** 密钥管理服务，安全存储数据库密码、API Key 等敏感信息。

**存了什么：**
- `rag-demo/pg-dsn`：完整的 PostgreSQL 连接串（含密码）
- `rag-demo/google-api-key`：Gemini API Key
- `rds!db-xxx`：RDS 自动管理的数据库密码

**工作流程：**
```
ECS 启动容器
    ↓
execution-role 从 Secrets Manager 读取密钥
    ↓
注入为容器环境变量（PG_DSN、GOOGLE_API_KEY）
    ↓
容器代码通过 os.environ 读取，密码不写在代码里
```

**为什么不直接写在代码或环境变量里：**
- 代码提交到 GitHub 会泄露
- Secrets Manager 有访问日志、自动轮换功能
- 符合最小权限原则

---

### 5. ECR（Elastic Container Registry）
**是什么：** AWS 的私有 Docker 镜像仓库，类似私有版 Docker Hub。

**仓库名：** `rag-demo`
**镜像地址：** `569260897196.dkr.ecr.eu-west-1.amazonaws.com/rag-demo:latest`

**工作流程：**
```
本地 / GitHub Actions
    ↓
docker build → 构建镜像
    ↓
aws ecr get-login-password → 登录 ECR
    ↓
docker push → 推送镜像到 ECR
    ↓
ECS 从 ECR 拉取镜像启动容器
```

**为什么用 ECR 而不是 Docker Hub：**
- 与 ECS 同在 AWS 内网，拉取速度快
- IAM 权限控制，更安全
- 与 GitHub Actions 集成简单

---

### 6. ECS Fargate（Elastic Container Service）
**是什么：** 容器运行服务。Fargate 模式下不需要管理服务器，AWS 自动分配计算资源。

**集群名：** `rag-demo-cluster`
**服务名：** `rag-demo-service`
**Task Definition：** `rag-demo:2`（第2版）

**核心概念：**
```
Cluster（集群）
  └── Service（服务，维持指定数量的 Task 运行）
        └── Task（任务，实际运行的容器实例）
              └── Container（容器，运行 FastAPI）
```

**Task Definition 配置：**
- CPU：1024（1 vCPU）
- 内存：2048 MB（2 GB）
- 镜像：ECR 中的 rag-demo:latest
- 环境变量：从 Secrets Manager 注入 PG_DSN、GOOGLE_API_KEY
- 日志：写入 CloudWatch `/ecs/rag-demo`

**Fargate vs EC2 模式：**
- Fargate：无服务器，按 Task 运行时间计费，无需管理底层 EC2
- EC2 模式：需要自己管理服务器，支持 GPU（g4dn）

---

### 7. Security Group（安全组）
**是什么：** 虚拟防火墙，控制进出流量。

**我们创建的规则：**

| 安全组 | 规则 | 作用 |
|--------|------|------|
| `rag-demo-ecs-sg` | 入站 TCP 8000，来源 0.0.0.0/0 | 允许外部访问 FastAPI |
| RDS 默认安全组 | 入站 TCP 5432，来源 rag-demo-ecs-sg | 只允许 ECS 容器访问数据库 |
| RDS 默认安全组 | 入站 TCP 5432，来源 你的 IP | 允许本地 psql 连接调试 |

**核心原则：** 最小权限，RDS 不直接暴露给公网，只允许 ECS 容器访问。

---

### 8. CloudWatch
**是什么：** AWS 的日志和监控服务。

**日志组：** `/ecs/rag-demo`

**收集什么：**
- FastAPI 的所有 stdout/stderr 输出
- 应用启动日志、请求日志、错误日志

**查看日志命令：**
```bash
aws logs get-log-events \
  --log-group-name /ecs/rag-demo \
  --log-stream-name "ecs/rag-api/<task-id>" \
  --region eu-west-1 \
  --limit 50 \
  --query "events[-20:].message" \
  --output text
```

---

### 9. GitHub Actions CI/CD
**是什么：** 代码推送后自动执行的流水线。

**配置文件：** `.github/workflows/deploy.yml`

**触发条件：** push 到 main 分支

**执行流程：**
```
git push origin main
    ↓
GitHub Actions 触发
    ↓
① actions/checkout — 拉取代码
    ↓
② configure-aws-credentials — 用 Secrets 里的 Access Key 登录 AWS
    ↓
③ amazon-ecr-login — 登录 ECR
    ↓
④ docker build + push — 构建新镜像，推送到 ECR（tag 为 commit SHA）
    ↓
⑤ aws ecs update-service --force-new-deployment — 触发 ECS 滚动更新
    ↓
ECS 用新镜像启动新容器，旧容器下线
```

**GitHub Secrets 配置：**
- `AWS_ACCESS_KEY_ID`：rag-developer 的 Access Key ID
- `AWS_SECRET_ACCESS_KEY`：rag-developer 的 Secret Access Key

---

### 10. EC2 g4dn.xlarge（待配额审批）
**是什么：** 带 GPU 的虚拟机，T4 GPU，16GB 显存。

**用途：** 运行 embedding server 和 reranker

**为什么需要 GPU：**
- BGE-M3 embedding 模型：查询时需要把文字转向量
- BGE-reranker-v2-m3：对召回的 top-50 结果精排
- CPU 推理太慢（reranker 单次 5-10s），GPU 加速到 <0.5s

**自动关机机制（省钱）：**
- EC2 上运行监控脚本，30 分钟无请求自动关机
- ECS 容器检测 embedding server 不在线时，调用 boto3 自动启动 EC2
- 只在 demo 时运行，按小时计费约 $0.16/小时（Spot 价格）

---

## 完整请求链路

```
用户发送查询
    ↓
ECS Fargate（FastAPI）接收请求
    ↓
① 检查 embedding server 是否在线（EC2）
   - 不在线 → boto3 启动 EC2 → 等待就绪
    ↓
② Query Embedding（发 HTTP 请求到 EC2:6006）
   BGE-M3 把查询文字转为向量
    ↓
③ 并行检索
   - pgvector（RDS）：向量相似度搜索 top-50
   - BM25（从 S3 加载的索引）：关键词匹配 top-50
    ↓
④ RRF 融合：合并两路结果，去重排序
    ↓
⑤ Reranker（EC2）：BGE cross-encoder 精排，取 top-5
    ↓
⑥ LangGraph Agent：把 top-5 chunks 交给 Gemini 生成答案
    ↓
⑦ Reflector：自我评估答案质量，不够好则重试（最多2次）
    ↓
返回答案 + 引用来源（SSE 流式）
```

---

## 费用估算（eu-west-1）

| 服务 | 规格 | 费用 |
|------|------|------|
| ECS Fargate | 1 vCPU / 2GB，持续运行 | ~$25/月 |
| RDS PostgreSQL | db.t3.micro，Free Tier | $0（首年） |
| EC2 g4dn.xlarge | Spot，每天 demo 2小时 | ~$10/月 |
| S3 | 存储 + 请求 | <$1/月 |
| Secrets Manager | 2 个密钥 | ~$1/月 |
| CloudWatch | 日志存储 | <$1/月 |
| **总计** | | **~$37/月** |

---

## 面试一句话总结

> "Deployed production RAG system on AWS — FastAPI on ECS Fargate with auto-deployment via GitHub Actions CI/CD; pgvector on RDS for vector storage; BGE-M3 embedding and reranker on EC2 g4dn (GPU); BM25 index persisted on S3; credentials managed via Secrets Manager."
