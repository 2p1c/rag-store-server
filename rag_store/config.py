import os
from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_ID = "sentence-transformers/msmarco-corpus"
DATASET_CONFIG = "passage"
DATASET_SPLIT = "train"
CORPUS_SIZE = 300_000

CANDIDATES = 20
MIN_SCORE = 0.35
MAX_CHARS = 6000

ENCODE_BATCH_SIZE = 64

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 中英翻译提示词：只改这一段。用户检索词会作为 user 消息另发，不要写进这里。
TRANSLATE_PROMPT = """You translate search queries from Chinese to English for an English document index.
Reply with only the English query. No quotes, no explanation, no extra words.
If the text is already English, repeat it unchanged."""


def _load_dotenv() -> None:
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

INDEX_DIR = os.environ.get("INDEX_DIR", ".indexes/msmarco-minilm")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8080"))


def deepseek_api_key() -> str:
    return os.environ.get("DEEPSEEK_API_KEY", "").strip()
