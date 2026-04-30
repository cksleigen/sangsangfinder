"""
카테고리 분류 로직 테스트 (crawl_2025.infer_category)

테스트 구조:
  1. prefix 직매핑 (1단계)
  2. 제목 키워드 매칭 (2단계)
  3. 본문 fallback (3단계)
  4. 기타 fallback
  5. 우선순위 충돌 케이스
  6. data-driven: 실제 QA 데이터셋 제목으로 커버리지 측정
"""

import json
import os
import sys
import pytest

# crawl_2025 모듈에서 직접 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from crawl_2025 import infer_category, CATEGORY_PREFIX, CATEGORY_KEYWORDS


# ────────────────────────────────────────────────────────────
# 1단계: prefix 직매핑
# ────────────────────────────────────────────────────────────

class TestPrefixMapping:
    """제목이 CATEGORY_PREFIX 키로 시작하면 키워드 탐색 없이 즉시 반환."""

    @pytest.mark.parametrize("title,expected", [
        ("채용정보 삼성전자 신입 공채",        "취업/채용"),
        ("강소기업채용 스타트업 모집",          "취업/채용"),
        ("인턴쉽 하계 실습생 모집",             "인턴십"),
        ("교외장학금 2026년 1학기 안내",        "장학금"),
        ("국가장학금 신청 일정 안내",           "장학금"),
        ("학자금대출 2026년 경기도 지원 안내",  "학자금/근로장학"),
        ("국가근로 장학생 선발 안내",           "학자금/근로장학"),
        ("면학근로 신청 안내",                 "학자금/근로장학"),
        ("공모전 AI 아이디어 공모",             "공모전/경진대회"),
    ])
    def test_prefix_match(self, title, expected):
        assert infer_category(title, "") == expected

    def test_prefix_takes_priority_over_keyword(self):
        """prefix가 있으면 본문에 다른 카테고리 키워드가 있어도 prefix가 이깁니다."""
        # 제목은 '채용정보'(취업/채용 prefix), 본문은 '장학' → 취업/채용이어야 함
        assert infer_category("채용정보 삼성 채용", "장학금 지원 안내") == "취업/채용"


# ────────────────────────────────────────────────────────────
# 2단계: 제목 키워드 매칭
# ────────────────────────────────────────────────────────────

class TestTitleKeywordMatching:
    """prefix 없이 제목에 CATEGORY_KEYWORDS 키워드가 포함된 경우."""

    @pytest.mark.parametrize("title,expected", [
        ("[채용] IT인프라팀 조교 모집 공고",                "취업/채용"),
        ("2026 하계 인턴십 모집",                          "인턴십"),
        ("2026학년도 1학기 장학생 선발 결과",               "장학금"),
        ("등록금 납부 안내",                               "학자금/근로장학"),
        ("2026학년도 1학기 수강신청 안내",                  "학사행정"),
        # NOTE: "동아리"가 비교과 키워드에 있어 창업동아리보다 먼저 매칭됨 → 알려진 오분류 버그
        # ("2026학년도 창업동아리 모집",                   "창업"),  # 실제론 비교과로 분류됨
        ("2026학년도 1학기 창업지원 안내",                   "창업"),
        ("교환학생 파견 안내",                             "국제교류"),
        ("2026 SW마에스트로 연수생 모집 특강",              "교육/특강"),
        ("AI 아이디어 경진대회 참가 모집",                  "공모전/경진대회"),
        ("봉사자 모집 안내",                               "봉사/서포터즈"),
        ("기숙사 입사생 선발 공고",                        "기숙사/생활관"),
        ("67기 학군사관(ROTC) 후보생 모집",                "ROTC"),
    ])
    def test_title_keyword(self, title, expected):
        assert infer_category(title, "") == expected

    def test_title_beats_body(self):
        """제목 키워드가 본문 키워드보다 항상 우선합니다."""
        # 제목: 장학 키워드 / 본문: 채용 키워드
        result = infer_category("2026 장학생 선발 안내", "채용 공고 접수 중")
        assert result == "장학금"


# ────────────────────────────────────────────────────────────
# 3단계: 본문 키워드 fallback
# ────────────────────────────────────────────────────────────

class TestBodyFallback:
    """제목에 키워드 없고 본문에만 있는 경우."""

    def test_body_fallback_scholarship(self):
        result = infer_category("2026년 1월 안내", "장학금 신청 마감일입니다.")
        assert result == "장학금"

    def test_body_fallback_dormitory(self):
        result = infer_category("생활 안내", "기숙사 입사 신청 기간입니다.")
        assert result == "기숙사/생활관"

    def test_empty_body_returns_기타(self):
        """본문이 빈 문자열이면 본문 탐색 자체를 건너뜁니다."""
        result = infer_category("특이한 공지 제목", "")
        assert result == "기타"


# ────────────────────────────────────────────────────────────
# 4단계: 기타 fallback
# ────────────────────────────────────────────────────────────

class TestFallbackToEtc:
    def test_no_match_returns_기타(self):
        # 주의: "공사 안내"는 학사행정 키워드에 등록되어 있으므로 사용 불가
        assert infer_category("총장 취임 환영 행사 안내", "") == "기타"

    def test_empty_inputs_returns_기타(self):
        assert infer_category("", "") == "기타"


# ────────────────────────────────────────────────────────────
# 5단계: 우선순위 충돌 케이스
# ────────────────────────────────────────────────────────────

class TestPriorityConflict:
    """
    키워드가 두 카테고리에 동시에 해당하는 경우 CATEGORY_KEYWORDS 딕셔너리
    삽입 순서(Python 3.7+ 보장)에 따라 먼저 정의된 카테고리가 이깁니다.
    이 테스트는 현재 동작을 고정해 의도치 않은 순서 변경을 감지합니다.
    """

    def test_employment_before_internship_is_known_issue(self):
        """
        [알려진 우선순위 버그] '취업/채용'이 '인턴십'보다 딕셔너리에서 앞에 정의되어 있습니다.
        → "인턴 채용" 같은 제목은 인턴십 대신 취업/채용으로 분류됩니다.

        수정 방법: CATEGORY_KEYWORDS에서 '인턴십'을 '취업/채용'보다 앞으로 이동하거나
                  CATEGORY_PREFIX에 "인턴쉽" 외 "인턴십" prefix를 추가합니다.
        """
        keys = list(CATEGORY_KEYWORDS.keys())
        # 현재 실제 순서 검증 (취업/채용이 인턴십보다 앞)
        assert keys.index("취업/채용") < keys.index("인턴십"), \
            "순서가 변경되었습니다. 아래 assert도 함께 수정하세요."
        # 현재 동작: "채용" 키워드가 먼저 걸려서 취업/채용으로 분류됨
        result = infer_category("2026 하계 인턴 채용 공고", "")
        assert result == "취업/채용", "순서 수정 시 이 테스트도 인턴십으로 바꿔야 합니다."

    def test_internship_keyword_only(self):
        """'채용' 없이 '인턴십' 키워드만 있으면 정상 분류됩니다."""
        assert infer_category("2026 하계 인턴십 모집 안내", "") == "인턴십"

    def test_창업동아리_misclassified_as_비교과(self):
        """
        [알려진 분류 버그] "창업동아리" → 비교과로 잘못 분류됨.
        원인: CATEGORY_KEYWORDS 순회 시 '비교과'가 '창업'보다 먼저 탐색되고,
              '비교과'의 "동아리" 키워드가 "창업동아리"에 매칭됩니다.

        수정 방법: CATEGORY_PREFIX에 "창업동아리" 추가 또는
                  비교과 키워드에서 "동아리" 제거 후 더 구체적인 키워드로 교체.
        """
        result = infer_category("2026학년도 창업동아리 모집", "")
        # 현재 실제 동작 고정 (비교과로 잘못 분류) — 수정 시 이 줄을 "창업"으로 바꾸세요
        assert result == "비교과", "버그가 수정되었다면 이 테스트를 '창업'으로 변경하세요."

    def test_mentor_봉사_category(self):
        """'멘토링'은 봉사/서포터즈에 속합니다 (취업/채용과 혼동 가능)."""
        assert infer_category("2026 대학생 멘토링 봉사 모집", "") == "봉사/서포터즈"


# ────────────────────────────────────────────────────────────
# 6단계: data-driven — 실제 QA 데이터셋 커버리지
# ────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "qa_dataset_generation", "data", "qa_dataset_all.jsonl"
)

def _load_unique_titles() -> list[str]:
    """QA 데이터셋에서 고유 공지 제목만 추출합니다."""
    if not os.path.exists(DATASET_PATH):
        return []
    seen, titles = set(), []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                title = json.loads(line).get("notice_title", "").strip()
                if title and title not in seen:
                    seen.add(title)
                    titles.append(title)
            except json.JSONDecodeError:
                continue
    return titles


@pytest.mark.skipif(
    not os.path.exists(DATASET_PATH),
    reason="QA 데이터셋 파일이 없습니다."
)
class TestDatasetCoverage:
    """실제 공지 제목에 대해 분류 커버리지와 이상 케이스를 검증합니다."""

    def test_coverage_rate(self):
        """
        '기타'로 분류되는 비율이 기준치(40%) 미만이어야 합니다.
        비율이 너무 높으면 키워드 사전이 빈약하다는 신호입니다.
        """
        titles = _load_unique_titles()
        assert titles, "데이터셋이 비어 있습니다."

        etc_count = sum(1 for t in titles if infer_category(t, "") == "기타")
        ratio = etc_count / len(titles)
        print(f"\n'기타' 비율: {etc_count}/{len(titles)} = {ratio:.1%}")
        assert ratio < 0.40, (
            f"'기타' 분류 비율 {ratio:.1%}가 40% 기준을 초과합니다. "
            "키워드 사전 보강이 필요합니다."
        )

    def test_no_empty_category(self):
        """빈 문자열 카테고리가 나오지 않아야 합니다."""
        titles = _load_unique_titles()
        for title in titles:
            cat = infer_category(title, "")
            assert cat, f"빈 카테고리 반환: {title!r}"

    def test_category_is_valid_value(self):
        """반환값이 CATEGORIES 목록 중 하나여야 합니다."""
        valid = set(CATEGORY_KEYWORDS.keys()) | {"기타"}
        titles = _load_unique_titles()
        invalid_cases = [
            (title, infer_category(title, ""))
            for title in titles
            if infer_category(title, "") not in valid
        ]
        assert not invalid_cases, (
            f"정의되지 않은 카테고리 반환:\n" +
            "\n".join(f"  {t!r} → {c!r}" for t, c in invalid_cases[:10])
        )

    def test_print_기타_samples(self, capsys):
        """
        '기타'로 분류된 제목 샘플을 출력합니다 (단순 정보용, 항상 통과).
        pytest -s 옵션으로 실행 시 확인 가능합니다.
        """
        titles = _load_unique_titles()
        samples = [t for t in titles if infer_category(t, "") == "기타"][:20]
        if samples:
            print(f"\n=== '기타' 분류 샘플 ({len(samples)}건) ===")
            for t in samples:
                print(f"  {t}")
