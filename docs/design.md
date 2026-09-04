# 向量库服务设计（本仓库范围）

日期：2026-09-04  
配对仓库：minimal-agent-ts（只做 HTTP 查询客户端）。

## 目标

实现一个可被 Agent 调用的 search 服务：接收 query 文本，返回已装盒的文档片段。入库、embedding、FAISS 都在这里，保证 query 与文档用同一模型。

不是每轮对话都检索。Agent 把检索做成工具；本服务不感知对话历史。

## 数据流

```
文档 ──► embedding（入库模型）──► L2 归一化 ──► FAISS IndexFlatIP

Agent POST { query }
        │
        ▼
同一 embedding 模型：query → vector → L2 归一化
        │
        ▼
FAISS 取 CANDIDATES 条
        │
        ▼
方案 B：丢掉 score < MIN_SCORE，按分数填满 MAX_CHARS
        │
        ▼
{ results: [{ text, source, score }] }
```

## 本仓库做

- 文档切块与入库
- 选定并固定一个 embedding 模型
- 建 / 读 FAISS `IndexFlatIP`
- `POST /search` 实现方案 B
- 空结果返回 `[]`，不编造

## 本仓库不做

- Agent 的 `rag_search` 工具、`/rag` 斜杠命令、markdown 拼装
- IVF / HNSW、rerank、MMR、parent-child 回填
- 把 k / 阈值 / 预算暴露给 Agent 或模型
- 检索鉴权、多知识库（v1）
- 在 Agent 进程里跑 embedding 或 FAISS

## 常量

见 [CONTRACT.md](CONTRACT.md)。`MIN_SCORE` 随 embedding 模型标定，不写死成永远正确的数。
