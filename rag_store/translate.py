from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import httpx

from rag_store.config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    TRANSLATE_PROMPT,
    deepseek_api_key,
)

HAN_RE = re.compile(r"[\u4e00-\u9fff]")
TranslatePost = Callable[..., Any]


class TranslateError(Exception):
    pass


def needs_translation(query: str) -> bool:
    return HAN_RE.search(query) is not None


def translate_query(query: str, post: TranslatePost | None = None) -> str:
    if not needs_translation(query):
        return query
    api_key = deepseek_api_key()
    if not api_key:
        raise TranslateError("DEEPSEEK_API_KEY is not set")
    url = DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"
    try:
        response = (post or httpx.post)(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": TRANSLATE_PROMPT},
                    {"role": "user", "content": query},
                ],
                "temperature": 0,
            },
            timeout=20.0,
        )
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"]
    except TranslateError:
        raise
    except Exception as e:
        raise TranslateError(f"DeepSeek translation failed: {e}") from e
    if not isinstance(text, str):
        raise TranslateError("DeepSeek translation returned empty content")
    translated = text.strip().strip('"').strip("'")
    if not translated:
        raise TranslateError("DeepSeek translation returned empty content")
    return translated
