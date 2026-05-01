"""
check_chroma.py — ChromaDB 임베딩 저장 상태 검증 스크립트

사용법:
    python check_chroma.py
"""

import os
import re
import sys
from collections import Counter, defaultdict

import chromadb

_BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트
CHROMA_DB_PATH = os.path.join(_BASE_DIR, "chroma_db")
COLLECTION_NAME = "hansung_notices"
REQUIRED_META_KEYS = {"title", "url", "date", "category"}
CHUNK_ID_PATTERN   = re.compile(r"^[0-9a-f]{32}_\d+$")  # {md5}_{chunk_idx}

SEP = "─" * 56


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check_db_exists() -> bool:
    section("1. DB 경로 확인")
    db_file = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"[FAIL] chroma_db 디렉토리 없음: {CHROMA_DB_PATH}")
        return False
    if not os.path.exists(db_file):
        print(f"[FAIL] chroma.sqlite3 없음: {db_file}")
        return False
    size_mb = os.path.getsize(db_file) / 1024 / 1024
    print(f"[OK]   경로: {CHROMA_DB_PATH}")
    print(f"[OK]   DB 크기: {size_mb:.2f} MB")
    return True


def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    raw = client.list_collections()
    # v0.6.0+: list_collections() returns str list; older: Collection object list
    collections = raw if raw and isinstance(raw[0], str) else [c.name for c in raw]
    section("2. 컬렉션 확인")
    print(f"       존재하는 컬렉션: {collections}")
    if COLLECTION_NAME not in collections:
        print(f"[FAIL] '{COLLECTION_NAME}' 컬렉션이 없습니다.")
        return None
    col = client.get_collection(name=COLLECTION_NAME)
    print(f"[OK]   컬렉션 '{COLLECTION_NAME}' 로드 성공")
    return col


def check_count(col) -> int:
    section("3. 저장 개수")
    count = col.count()
    if count == 0:
        print("[FAIL] 저장된 데이터가 0개입니다. 임베딩을 먼저 실행하세요.")
    else:
        print(f"[OK]   총 청크 수: {count:,}개")
    return count


def check_sample_metadata(col):
    section("4. 샘플 메타데이터 검증 (처음 10개 청크)")
    data = col.get(limit=10, include=["documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]

    bad_ids, bad_metas, empty_docs = [], [], []
    for cid, doc, meta in zip(ids, docs, metas):
        if not CHUNK_ID_PATTERN.match(cid):
            bad_ids.append(cid)
        missing = REQUIRED_META_KEYS - set(meta.keys())
        if missing:
            bad_metas.append((cid, missing))
        if not doc or not doc.strip():
            empty_docs.append(cid)

    if bad_ids:
        print(f"[WARN] ID 형식 불일치 ({len(bad_ids)}개): {bad_ids[:3]}")
    else:
        print(f"[OK]   ID 형식 정상 (모두 {{md5}}_{{index}} 패턴)")

    if bad_metas:
        for cid, missing in bad_metas[:3]:
            print(f"[FAIL] {cid}: 메타데이터 누락 필드 = {missing}")
    else:
        print(f"[OK]   메타데이터 필수 필드 완전 (title, url, date, category)")

    if empty_docs:
        print(f"[WARN] 빈 document 청크 ({len(empty_docs)}개): {empty_docs[:3]}")
    else:
        print(f"[OK]   document 내용 정상")

    # 샘플 출력
    print(f"\n  --- 샘플 (첫 3개) ---")
    for cid, doc, meta in zip(ids[:3], docs[:3], metas[:3]):
        print(f"  ID      : {cid}")
        print(f"  제목    : {meta.get('title','(없음)')[:50]}")
        print(f"  날짜    : {meta.get('date','(없음)')}")
        print(f"  카테고리: {meta.get('category','(없음)')}")
        print(f"  doc길이 : {len(doc)}자")
        print()


def check_embedding_dim(col):
    section("5. 임베딩 차원 확인")
    data = col.get(limit=1, include=["embeddings"])
    embeddings = data.get("embeddings")
    if embeddings is None or len(embeddings) == 0 or embeddings[0] is None:
        print("[FAIL] 임베딩 벡터를 가져올 수 없습니다.")
        return
    dim = len(embeddings[0])
    # BM-K/KoSimCSE-roberta-multitask 기본 출력 차원: 768
    expected = 768
    if dim == expected:
        print(f"[OK]   임베딩 차원: {dim} (BM-K/KoSimCSE-roberta-multitask 기본값과 일치)")
    else:
        print(f"[WARN] 임베딩 차원: {dim} (기대값 {expected}와 다름 — 파인튜닝 모델 사용 중일 수 있음)")


def check_category_distribution(col, count: int):
    section("6. 카테고리 분포")
    # 전체를 한 번에 가져오면 메모리 부담 → limit으로 샘플링
    sample_size = min(count, 2000)
    data = col.get(limit=sample_size, include=["metadatas"])
    cat_counter = Counter(m.get("category", "(없음)") for m in data["metadatas"])
    total = sum(cat_counter.values())
    for cat, cnt in sorted(cat_counter.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / total * 30)
        print(f"  {cat:<20} {cnt:>5}개  {bar}")
    if sample_size < count:
        print(f"\n  (전체 {count:,}개 중 {sample_size:,}개 샘플 기준)")


def check_duplicate_docs(col, count: int):
    section("7. 중복 공지 URL 확인")
    sample_size = min(count, 2000)
    data = col.get(limit=sample_size, include=["metadatas"])
    url_chunks: dict[str, list] = defaultdict(list)
    for cid, meta in zip(data["ids"], data["metadatas"]):
        url_chunks[meta.get("url", "")].append(cid)

    multi = {url: ids for url, ids in url_chunks.items() if len(ids) > 1}
    print(f"[OK]   고유 공지 URL 수: {len(url_chunks):,}개")
    print(f"[OK]   멀티-청크 공지 수: {len(multi):,}개 (본문이 길어 분할된 것은 정상)")

    # 청크가 비정상적으로 많은 공지 (20개 초과) 경고
    outliers = {url: ids for url, ids in multi.items() if len(ids) > 20}
    if outliers:
        print(f"[WARN] 청크가 20개 초과인 공지 ({len(outliers)}개):")
        for url, ids in list(outliers.items())[:3]:
            print(f"       {url[:60]}  →  {len(ids)}청크")


def check_query(col):
    section("8. 벡터 검색 테스트")
    try:
        from sentence_transformers import SentenceTransformer
        base_model = "BM-K/KoSimCSE-roberta-multitask"
        embed_model_path = os.path.join(_BASE_DIR, "models", "embed_finetuned")
        model_path = embed_model_path if os.path.exists(embed_model_path) else base_model
        print(f"       임베딩 모델 로드: {model_path}")
        model = SentenceTransformer(model_path, device="cpu")
    except ImportError:
        print("[SKIP] sentence_transformers 미설치 — 쿼리 테스트를 건너뜁니다.")
        return

    queries = ["장학금 신청 기간", "취업박람회", "수강신청"]
    n = min(3, col.count())
    for q in queries:
        emb = model.encode(q).tolist()
        results = col.query(
            query_embeddings=[emb],
            n_results=n,
            include=["metadatas", "distances"],
        )
        top_meta = results["metadatas"][0][0]
        top_dist = results["distances"][0][0]
        top_sim  = round(1 - top_dist, 4)
        print(f"\n  쿼리: '{q}'")
        print(f"    → 제목     : {top_meta.get('title','')[:60]}")
        print(f"    → 카테고리 : {top_meta.get('category','')}")
        print(f"    → 유사도   : {top_sim}")
        if top_sim < 0.3:
            print(f"    [WARN] 유사도가 낮습니다 (< 0.3). 임베딩 품질을 확인하세요.")


def main():
    print(f"\n{'═'*56}")
    print(f"  ChromaDB 임베딩 검증 — {COLLECTION_NAME}")
    print(f"{'═'*56}")

    if not check_db_exists():
        sys.exit(1)

    col = load_collection()
    if col is None:
        sys.exit(1)

    count = check_count(col)
    if count == 0:
        sys.exit(1)

    check_sample_metadata(col)
    check_embedding_dim(col)
    check_category_distribution(col, count)
    check_duplicate_docs(col, count)
    check_query(col)

    section("결과 요약")
    print(f"  총 청크 수  : {count:,}개")
    print(f"  DB 위치     : {CHROMA_DB_PATH}")
    print(f"\n  검증 완료.")


if __name__ == "__main__":
    main()
