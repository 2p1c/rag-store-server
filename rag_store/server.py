from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_store.config import HOST, INDEX_DIR, PORT
from rag_store.store import VectorStore
from rag_store.translate import TranslateError


class SearchRequest(BaseModel):
    query: str


def create_app(store: VectorStore | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store if store is not None else VectorStore.load(INDEX_DIR)
        print(app.state.store.format_size())
        yield

    app = FastAPI(title="rag-vector-store", lifespan=lifespan)

    @app.post("/search")
    def search(req: SearchRequest) -> dict:
        query = req.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query must be a non-empty string")
        try:
            return {"results": app.state.store.search(query)}
        except TranslateError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("rag_store.server:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
