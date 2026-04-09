"""
QA Dataset Generation Pipeline (Teacher-Judge)
전략: Teacher-only, 유형별 강제 커버리지, Hard Negative, Notice-level Split
"""

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from anthropic import Anthropic

# 프로젝트 루트의 .env 로드 (python-dotenv 없이 직접 파싱)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

client = Anthropic()

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
TEACHER_MODEL   = "claude-sonnet-4-6"
JUDGE_MODEL     = "claude-haiku-4-5"

QA_PER_NOTICE           = 5  # deadline / eligibility / procedural / factual / 자유
UNANSWERABLE_PER_NOTICE = 1  # round(5 * 0.2 / 0.8) = 1
QUALITY_THRESHOLD       = 3
HARD_NEG_PER_NOTICE     = 3  # 동일 카테고리 내 hard negative 수

# TRAIN split — used for fine-tuning
# VAL/TEST split  — held out, touched only in final eval
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED  = 42

NOTICES_CACHE_PATH = "./data/notices_cache.json"
OUTPUT_PATH        = "./data/qa_dataset.jsonl"


# ─────────────────────────────────────────
# Step 1. Teacher로 QA 생성 (유형별 강제)
# ─────────────────────────────────────────
TEACHER_SYSTEM = """당신은 대학교 공지사항 기반 QA 데이터셋을 생성하는 전문가입니다.
공지사항을 분석해 학생들이 실제로 물어볼 법한 질문과 정확한 답변을 생성하세요.

반드시 JSON 배열만 출력하세요. 다른 텍스트는 절대 포함하지 마세요."""

TEACHER_PROMPT = """다음 공지사항을 읽고 QA 쌍 정확히 {n}개를 생성하세요.

━━ 유형별 강제 커버리지 (순서대로 1개씩) ━━
1. deadline   : 마감일, 신청 기한, 일정 관련 질문
2. eligibility: 신청 자격, 대상, 조건 관련 질문
3. procedural : 신청 방법, 절차, 제출 서류 관련 질문
4. factual    : 장소, 금액, 인원, 주관 기관 등 기타 사실 질문
5. (자유)     : 위 4가지로 담지 못한 공지의 핵심 정보 — 위 유형 중 가장 근접한 type으로 표기

공지에 특정 유형 정보가 없으면 가장 근접한 다른 유형으로 대체하되, 반드시 5개를 채우세요.

━━ 난이도 분포 (균등) ━━
- easy  : 공지 한 문장에 답이 그대로 있음
- medium: 공지 여러 문장을 조합해야 답이 나옴
- hard  : 추론 필요 또는 답의 표현이 질문과 다르게 paraphrase됨

━━ 오염 방지 (필수) ━━
- 질문에 답의 핵심 키워드를 그대로 포함하지 않는다
- 나쁜 예) "수강신청 정정 기간은 2월 3일인가요?" — 날짜 노출
- 좋은 예) "수강신청 정정은 언제까지 가능한가요?"
- 학생이 실제로 물어볼 법한 자연스러운 표현 사용

공지사항:
{notice}

출력 형식 (JSON 배열):
[
  {{
    "question": "질문",
    "answer": "공지사항에 근거한 정확한 답변",
    "type": "deadline|eligibility|procedural|factual",
    "difficulty": "easy|medium|hard",
    "source_span": "답변 근거가 되는 원문 구절 (20자 이내)"
  }}
]"""


def _parse_json_response(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ─────────────────────────────────────────
# Step 1-b. Unanswerable QA 생성
# ─────────────────────────────────────────
UNANSWERABLE_TEACHER_PROMPT = """다음 공지사항을 읽고, 공지사항만으로는 답할 수 없는 질문 {n}개를 생성하세요.

━━ 조건 ━━
- 공지사항 주제와 관련 있어 학생이 실제로 물어볼 법한 질문이어야 합니다
- 하지만 답변에 필요한 정보가 공지사항에 명시되어 있지 않아야 합니다
- type은 반드시 "unanswerable"
- difficulty는 easy·medium·hard 균등 분포

공지사항:
{notice}

출력 형식 (JSON 배열):
[
  {{
    "question": "질문",
    "answer": "해당 정보는 공지사항에 포함되어 있지 않습니다.",
    "type": "unanswerable",
    "difficulty": "easy|medium|hard",
    "source_span": null
  }}
]"""


def generate_unanswerable_qa(notice: str, n: int = UNANSWERABLE_PER_NOTICE) -> list[dict]:
    """Teacher 모델로 unanswerable QA 생성"""
    with client.messages.stream(
        model=TEACHER_MODEL,
        max_tokens=500,
        system=TEACHER_SYSTEM,
        messages=[{"role": "user", "content": UNANSWERABLE_TEACHER_PROMPT.format(n=n, notice=notice)}],
    ) as stream:
        final = stream.get_final_message()
    text = next(b.text for b in final.content if b.type == "text")
    items = _parse_json_response(text)
    for item in items:
        item["answerable"] = False
    return items


def generate_qa(notice: str, n: int = QA_PER_NOTICE) -> list[dict]:
    """Teacher 모델로 유형별 강제 커버리지 QA 생성"""
    with client.messages.stream(
        model=TEACHER_MODEL,
        max_tokens=2000,
        system=TEACHER_SYSTEM,
        messages=[{"role": "user", "content": TEACHER_PROMPT.format(n=n, notice=notice)}],
    ) as stream:
        final = stream.get_final_message()
    text = next(b.text for b in final.content if b.type == "text")
    items = _parse_json_response(text)
    for item in items:
        item["answerable"] = True
    return items


# ─────────────────────────────────────────
# Step 2. LLM-as-Judge 품질 검증 (단일 단계)
#   - Hallucination 탐지
#   - 품질 점수
# ─────────────────────────────────────────
JUDGE_SYSTEM = """당신은 QA 데이터셋 품질 평가 전문가입니다.
반드시 JSON만 출력하세요."""

JUDGE_EVAL_PROMPT = """공지사항과 아래 QA 쌍을 함께 평가하세요.

공지사항:
{notice}

Q: {question}
A: {answer}

판별 1. hallucination 여부:
- answer에 공지사항에 없는 사실·수치·날짜·조건이 포함되어 있으면 hallucination
- question이 공지사항 범위를 완전히 벗어난 내용이면 hallucination
- 공지사항에 명시되지 않은 내용을 answer가 추론·가정으로 채우면 hallucination

판별 2. 품질 점수:
- 5점: 공지사항 근거 명확, 답변 완전하고 정확
- 4점: 근거 있음, 답변 대체로 정확
- 3점: 근거 있으나 답변 불완전하거나 모호
- 2점: 근거 불명확하거나 답변 부정확
- 1점: 공지사항과 무관하거나 오답

JSON으로만 출력:
{{
  "hallucination": true|false,
  "hallucinated_span": "공지에 없는 부분 (없으면 null)",
  "hall_reason": "한 줄 이유",
  "score": 1~5,
  "score_reason": "한 줄 이유"
}}"""


def judge_qa(notice: str, qa: dict) -> dict:
    """LLM-as-Judge 단일 단계: hallucination 탐지 + 품질 점수"""
    with client.messages.stream(
        model=JUDGE_MODEL,
        max_tokens=350,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": JUDGE_EVAL_PROMPT.format(
            notice=notice,
            question=qa["question"],
            answer=qa["answer"],
        )}],
    ) as stream:
        final = stream.get_final_message()
    eval_text = next(b.text for b in final.content if b.type == "text")
    eval_result = _parse_json_response(eval_text)

    hall_reason = eval_result.get("hall_reason")
    is_hallucinated = bool(eval_result.get("hallucination"))
    score_reason = eval_result.get("score_reason")
    score = eval_result.get("score", 0)

    return {
        **qa,
        "hallucination": is_hallucinated,
        "hallucinated_span": eval_result.get("hallucinated_span") if is_hallucinated else None,
        "hall_reason": hall_reason if is_hallucinated else None,
        "judge_score": 0 if is_hallucinated else score,
        "judge_reason": f"[HALLUCINATION] {hall_reason}" if is_hallucinated else score_reason,
    }


# ─────────────────────────────────────────
# Step 3. Hard Negative 할당
#   동일 카테고리 내 다른 공지를 hard negative로 지정
# ─────────────────────────────────────────
def assign_hard_negatives(all_qa: list[dict], notices: list[dict]) -> list[dict]:
    """
    각 QA에 hard_negative_chunk_ids 필드 추가.
    동일 카테고리 내 다른 공지 중 최대 HARD_NEG_PER_NOTICE개를 샘플링.
    """
    # 카테고리 → notice_id 목록 매핑
    category_to_ids: dict[str, list[int]] = defaultdict(list)
    for i, notice in enumerate(notices):
        category_to_ids[notice.get("category", "기타")].append(i)

    rng = random.Random(SPLIT_SEED)

    for qa in all_qa:
        notice_id = qa["notice_id"]
        category = notices[notice_id].get("category", "기타")
        candidates = [nid for nid in category_to_ids[category] if nid != notice_id]
        sampled = rng.sample(candidates, min(HARD_NEG_PER_NOTICE, len(candidates)))
        qa["hard_negative_chunk_ids"] = [f"notice_{nid}" for nid in sampled]

    return all_qa


# ─────────────────────────────────────────
# Step 4. Notice 단위 Split
#   QA 단위가 아닌 공지 단위로 train/val/test 분리
#   → paraphrase 오염 방지
# ─────────────────────────────────────────
def assign_splits(all_qa: list[dict], notices: list[dict]) -> list[dict]:
    """
    # TRAIN split — used for fine-tuning
    # VAL/TEST split  — held out, touched only in final eval
    공지(notice) 단위로 split을 나누고 각 QA에 split 필드를 추가.
    """
    notice_ids = list(range(len(notices)))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(notice_ids)

    n = len(notice_ids)
    n_train = int(n * SPLIT_RATIO["train"])
    n_val   = int(n * SPLIT_RATIO["val"])

    train_ids = set(notice_ids[:n_train])
    val_ids   = set(notice_ids[n_train:n_train + n_val])
    test_ids  = set(notice_ids[n_train + n_val:])

    split_map = {**{i: "train" for i in train_ids},
                 **{i: "val"   for i in val_ids},
                 **{i: "test"  for i in test_ids}}

    for qa in all_qa:
        qa["split"] = split_map[qa["notice_id"]]

    train_count = sum(1 for qa in all_qa if qa["split"] == "train")
    val_count   = sum(1 for qa in all_qa if qa["split"] == "val")
    test_count  = sum(1 for qa in all_qa if qa["split"] == "test")
    print(f"\n[Split] train={train_count} / val={val_count} / test={test_count}  "
          f"(공지 기준: train={len(train_ids)} / val={len(val_ids)} / test={len(test_ids)})")

    return all_qa


# ─────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────
def format_notice(n: dict) -> str:
    return f"제목: {n['title']}\n날짜: {n['date']}\n카테고리: {n['category']}\n\n{n['body']}"


# ─────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────
def run_pipeline(notices: list[dict], output_path: str = OUTPUT_PATH):
    all_qa = []
    stats = {"total": 0, "passed": 0, "hallucinated": 0, "low_quality": 0, "unanswerable": 0}

    for i, notice_obj in enumerate(notices):
        notice_str = format_notice(notice_obj)
        print(f"\n[{i+1}/{len(notices)}] {notice_obj['title'][:50]}")

        # Step 1: Teacher QA 생성 (유형별 강제)
        print(f"  → Teacher(Sonnet 4.6): QA {QA_PER_NOTICE}개 생성 중 (deadline/eligibility/procedural/factual/자유)...")
        try:
            qa_list = generate_qa(notice_str)
            print(f"     {len(qa_list)}개 생성 완료")
        except Exception as e:
            print(f"     ⚠️ QA 생성 실패: {e}")
            continue
        time.sleep(0.5)

        # Step 1-b: Unanswerable QA 생성
        print(f"  → Teacher(Sonnet 4.6): unanswerable QA {UNANSWERABLE_PER_NOTICE}개 생성 중...")
        try:
            unanswerable_qa = generate_unanswerable_qa(notice_str)
            print(f"     unanswerable {len(unanswerable_qa)}개 생성 완료")
        except Exception as e:
            print(f"     ⚠️ unanswerable 생성 실패: {e}")
            unanswerable_qa = []
        time.sleep(0.5)

        # Step 2: Judge 검증
        print("  → Judge(Haiku 4.5): hallucination + 품질 점수 단일 평가 중...")
        for qa in qa_list:
            qa["notice_id"] = i
            qa["notice_title"] = notice_obj["title"]
            qa["source_chunk_id"] = f"notice_{i}"
            try:
                judged = judge_qa(notice_str, qa)
            except Exception as e:
                print(f"     ⚠️ judge 실패: {e}")
                continue

            stats["total"] += 1
            if judged["hallucination"]:
                stats["hallucinated"] += 1
                print(f"     [HALL] {qa['question'][:40]} → {judged['hallucinated_span']}")
            elif judged["judge_score"] >= QUALITY_THRESHOLD:
                all_qa.append(judged)
                stats["passed"] += 1
            else:
                stats["low_quality"] += 1
            time.sleep(0.3)

        # Unanswerable QA는 Judge 없이 바로 수집 (의도적으로 답 없는 질문)
        for qa in unanswerable_qa:
            qa["notice_id"] = i
            qa["notice_title"] = notice_obj["title"]
            qa["source_chunk_id"] = f"notice_{i}"
            qa.setdefault("hallucination", False)
            qa.setdefault("hallucinated_span", None)
            qa.setdefault("hall_reason", None)
            qa.setdefault("judge_score", None)
            qa.setdefault("judge_reason", None)
            all_qa.append(qa)
            stats["unanswerable"] += 1

        print(f"     통과: {stats['passed']} / unanswerable: {stats['unanswerable']} / "
              f"hallucination: {stats['hallucinated']} / 저품질: {stats['low_quality']} / 전체: {stats['total']}")

    # Step 3: Hard Negative 할당
    print("\n→ Hard Negative 할당 중 (동일 카테고리 내 공지)...")
    all_qa = assign_hard_negatives(all_qa, notices)

    # Step 4: Notice 단위 Split
    all_qa = assign_splits(all_qa, notices)

    # 저장
    out = Path(output_path)
    with out.open("w", encoding="utf-8") as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    total_saved = stats["passed"] + stats["unanswerable"]
    print(f"\n{'='*50}")
    print(f"완료! 총 {total_saved}개 QA 저장 → {output_path}")
    if total_saved:
        print(f"  Answerable:    {stats['passed']} ({stats['passed']/total_saved*100:.1f}%)")
        print(f"  Unanswerable:  {stats['unanswerable']} ({stats['unanswerable']/total_saved*100:.1f}%)")
    if stats["total"] > 0:
        print(f"  Hallucination: {stats['hallucinated']}/{stats['total']} ({stats['hallucinated']/stats['total']*100:.1f}%)")
        print(f"  저품질 탈락:   {stats['low_quality']}/{stats['total']} ({stats['low_quality']/stats['total']*100:.1f}%)")
    return all_qa


if __name__ == "__main__":
    data = json.load(open(NOTICES_CACHE_PATH, encoding="utf-8"))
    notices = [n for n in data if len(n.get("body", "")) > 300]
    print(f"선택된 공지 {len(notices)}건 (body > 300자)")
    run_pipeline(notices)
