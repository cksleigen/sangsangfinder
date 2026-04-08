import json
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import feedparser

RSS_URL = "https://www.hansung.ac.kr/bbs/hansung/2127/rssList.do?row=50"
STATE_PATH = Path("state/seen_items.json")
MAX_SEEN_ITEMS = 500


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"seen_ids": [], "last_checked": None}

    with STATE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def clean_url(url: str) -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    params.pop("layout", None)
    new_query = urlencode({key: value[0] for key, value in params.items()})
    return urlunparse(parsed._replace(query=new_query))


def format_date(entry) -> str:
    published = entry.get("published") or entry.get("updated")
    if not published:
        return datetime.now().strftime("%Y.%m.%d")

    try:
        return parsedate_to_datetime(published).strftime("%Y.%m.%d")
    except (TypeError, ValueError, OverflowError):
        return published


def get_entry_id(entry) -> str | None:
    return entry.get("id") or entry.get("guid") or entry.get("link")


def fetch_rss_items() -> list[dict]:
    feed = feedparser.parse(RSS_URL)

    if feed.bozo:
        raise RuntimeError(f"RSS 파싱 실패: {feed.bozo_exception}")

    items = []
    for entry in feed.entries:
        entry_id = get_entry_id(entry)
        link = entry.get("link")
        title = entry.get("title")

        if not entry_id or not link or not title:
            continue

        items.append(
            {
                "entry_id": entry_id,
                "title": title.strip(),
                "url": clean_url(link),
                "date": format_date(entry),
            }
        )

    return items


def pick_new_items(items: list[dict], seen_ids: set[str]) -> list[dict]:
    new_items = [item for item in items if item["entry_id"] not in seen_ids]
    new_items.reverse()
    return new_items


def build_next_state(previous_state: dict, items: list[dict]) -> dict:
    latest_ids = [item["entry_id"] for item in items]
    combined_ids = latest_ids + previous_state.get("seen_ids", [])

    deduped_ids = []
    seen = set()
    for entry_id in combined_ids:
        if entry_id in seen:
            continue
        seen.add(entry_id)
        deduped_ids.append(entry_id)

    return {
        "seen_ids": deduped_ids[:MAX_SEEN_ITEMS],
        "last_checked": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    print("RSS 기반 신규 공지 확인 시작")
    state = load_state()
    items = fetch_rss_items()

    if not items:
        print("RSS 항목이 없습니다.")
        save_state(build_next_state(state, items))
        return 0

    seen_ids = set(state.get("seen_ids", []))
    new_items = pick_new_items(items, seen_ids)

    if not seen_ids:
        print(f"초기 실행 감지: RSS 최근 {len(items)}건을 기준 상태로 저장합니다.")
        save_state(build_next_state(state, items))
        return 0

    if not new_items:
        print("새 공지가 없습니다.")
        save_state(build_next_state(state, items))
        return 0

    print(f"새 공지 {len(new_items)}건 발견")
    for item in new_items:
        print(f"  - [{item['date']}] {item['title']}")

    from embedding_search_v1 import index_notices

    index_notices(
        [
            {
                "title": item["title"],
                "url": item["url"],
                "date": item["date"],
            }
            for item in new_items
        ]
    )

    save_state(build_next_state(state, items))
    print("상태 파일 저장 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
