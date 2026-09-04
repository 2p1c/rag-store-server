# 检索契约

Agent 只发 query 文本。embedding、FAISS、过滤和装盒都在本服务完成。

## HTTP

```
POST /search
Content-Type: application/json

{"query": "检索词"}
```

成功 200：

```json
{
  "results": [
    { "text": "chunk 原文", "source": "doc.md", "score": 0.82 }
  ]
}
```

| 字段 | 要求 |
|---|---|
| `text` | 文档片段。空或缺失的条目会被 Agent 丢掉 |
| `source` | 来源标识 |
| `score` | cosine，越大越相关 |
| 没有命中 | `{"results": []}`，不要编造文档 |
| 非 2xx | Agent 把 `HTTP {status}` 当作工具错误 |

请求体**只有** `query`。不要让 Agent 或模型传 k、阈值、预算。

v1 无鉴权，默认本机或内网。Agent 超时 20 秒。

## Embedding

Query 必须用**入库时同一个** embedding 模型变成向量。不要在 Agent 里再调 embedding API。

换模型等于重建索引。

## FAISS

- 用 `IndexFlatIP`（暴力精确检索）。个人知识库不上 IVF / HNSW。
- 入库和查询前都做 L2 归一化，内积即 cosine。

## 方案 B（写死在本服务）

先取候选，再按分数过滤，再按字符预算装盒：

| 常量 | 建议默认 | 作用 |
|---|---|---|
| `CANDIDATES` | 20 | FAISS 先取多少条 |
| `MIN_SCORE` | 0.35 | cosine 低于此丢弃（按实际模型再调） |
| `MAX_CHARS` | 6000 | 按分数从高到低装盒的字符预算 |

一条都过不了阈值时返回空 `results`。

## Agent 侧用法（不在本仓库实现）

- 模型需要个人/领域文档时自己调 `rag_search`。
- 用户也可发 `/rag 检索词` 强制本轮检索（只认小写 `/rag`）。
- 时事和公开网页走 `web_search`，不走本服务。

## 本地冒烟

```bash
curl -s -X POST http://127.0.0.1:8080/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"test"}'
```

应返回带 `results` 数组的 JSON。然后启动 Agent，发：`/rag test`。
