import re
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

from .config import (
    CATEGORY_PATTERN, SUFFIX_PATTERN,
    CATEGORY_PREFIX, CATEGORY_KEYWORDS,
    CHUNK_SIZE, CHUNK_OVERLAP,
)


def clean_url(url: str) -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    params.pop("layout", None)
    new_query = urlencode({k: v[0] for k, v in params.items()})
    return urlunparse(parsed._replace(query=new_query))


def clean_title(raw: str) -> str:
    title = raw.replace("\n", " ").replace("\r", " ")
    title = re.sub(r"\s{2,}", " ", title).strip()
    title = CATEGORY_PATTERN.sub("", title).strip()
    title = SUFFIX_PATTERN.sub("", title).strip()
    return title


def infer_category(title: str, body: str) -> str:
    for prefix, cat in CATEGORY_PREFIX.items():
        if title.startswith(prefix):
            return cat
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in title for kw in keywords):
            return cat
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in body for kw in keywords):
            return cat
    return "기타"


def tokenize_ko(text: str) -> list[str]:
    return re.findall(r"[\w가-힣]+", text.lower())


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + CHUNK_SIZE])
        if start + CHUNK_SIZE >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks
