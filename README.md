# rag-vector-store

给 [minimal-agent-ts](https://github.com/2p1c/my-minimal-agent) 用的外接向量库。

**本仓库负责：** 文档入库、embedding、FAISS 索引、按方案 B 检索并返回 JSON。

**Agent 仓库负责：** `POST` 一段 query，把结果当成 `rag_search` 工具输出。它不调 embedding，也不跑 FAISS。

当前只有指导文档，还没有服务代码。实现时按 `docs/CONTRACT.md` 对齐 HTTP；背景和取舍见 `docs/design.md`。

## 文档

| 文件 | 内容 |
|---|---|
| [docs/CONTRACT.md](docs/CONTRACT.md) | Search HTTP、FAISS、同一 embedding、方案 B 常量、Agent 怎么接 |
| [docs/design.md](docs/design.md) | 已确认的职责切分与非目标 |

## 和 Agent 怎么连

1. 本服务监听例如 `http://127.0.0.1:8080/search`。
2. Agent 的 `.env` 写完整 URL（不要只写 host）：

   ```
   RAG_SEARCH_URL=http://127.0.0.1:8080/search
   ```

3. 重启 Agent。未配置则不注册 `rag_search`。
