# ============================================================
# crawl_2025_titles.py — 2025년 한성대 공지 제목·URL·날짜·카테고리 수집
#
# 실행:
#   python crawl_2025_titles.py
#
# 출력:
#   crawl_2025_titles.json  ← 기존 파일 있으면 중복 건너뜀
# ============================================================

import json
import os
import time

import requests
from bs4 import BeautifulSoup

HEADERS  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
BASE_URL = "https://www.hansung.ac.kr"
BOARD_ID = "2127"
LIST_URL = f"{BASE_URL}/bbs/hansung/{BOARD_ID}/artclList.do"
OUT_PATH = "crawl_2025_titles.json"

DELAY    = 0.3   # 요청 간 딜레이(초)
YEAR     = "2025"

# ── 카테고리 분류 ──────────────────────────────────────────────

CATEGORY_PREFIX = {
    "채용정보":    "취업/채용",
    "강소기업채용": "취업/채용",
    "인턴쉽":     "인턴십",
    "교외장학금":  "장학금",
    "국가장학금":  "장학금",
    "학자금대출":  "학자금/근로장학",
    "국가근로":    "학자금/근로장학",
    "면학근로":    "학자금/근로장학",
    "공모전":     "공모전/경진대회",
}

CATEGORY_KEYWORDS = {
    "ROTC":           ["ROTC", "학군사관", "학군단", "현역병 모집", "현역병모집", "예비군", "전문사관",
                       "재병역판정검사"],
    "기숙사/생활관":   ["기숙사", "생활관", "상상빌리지", "우촌학사", "임대기숙사", "사감",
                       "입사생 선발", "대학생주택", "학사관"],
    "비교과":         ["비교과", "동아리", "D-School", "포럼", "대동제", "영상제", "입학식",
                       "HS CREW", "상상파크", "라이프 디자인", "문화탐방", "만우절", "오찬 소통",
                       "Lunch with", "천원의 아침밥", "ESG", "진로집단상담", "리더십 탐험",
                       "학생축제", "문화제", "페스티벌", "소모임",
                       "디즈니 프로그램", "디즈니프로그램", "FSU-Disney", "Disney",
                       "새내기 새로배움터", "새로배움터", "총학생회",
                       "사진전", "영상·사진전", "진로 설명회", "트랙 진로",
                       # 지역 청년 프로그램·학생 복지
                       "학생자치기구",
                       "마음건강",
                       "건강검진", "건강 검진",
                       "대사증후군",
                       "건강관리실",
                       "아웃리치",
                       # 지역 연계 청년공간·행사
                       "동선이음",
                       "청년의 날",
                       "청년문화예술패스",
                       "청년정책 네트워크", "청년정책네트워크",
                       "온통청년",
                       "소셜다이닝",
                       "통일체험",
                       "영테크",
                       "리빙랩"],
    "취업/채용":      ["채용", "신입", "공채", "취업", "채용박람회", "취업박람회", "모집공고",
                       "직무", "채용연계", "일반채용", "추천채용",
                       "초빙 공고", "조교 모집", "수습직원"],
    "인턴십":         ["인턴", "인턴십", "일경험", "체험형", "현장실습", "IPP"],
    "장학금":         ["장학", "장학생", "장학재단", "장학금", "기부장학", "장학사업", "스칼라십", "장학지원"],
    "학자금/근로장학": ["학자금대출", "학자금", "이자지원", "국가근로", "면학근로", "근로장학", "대출이자",
                       "등록금 납부", "등록금 분할", "학기초과자 등록금",
                       "사회첫출발", "청년월세지원", "이사비 지원", "중개보수"],
    "학사행정":       ["수강신청", "수강정정", "졸업", "휴학", "복학", "학점", "트랙변경", "성적",
                       "폐강", "복수전공", "부전공", "휴복학", "재입학", "연계전공",
                       "Micro Degree", "MD과정", "교양영어", "이수신청", "트랙선택",
                       "계절학기", "수업평가", "학위취득유예", "수강포기", "서면신청", "서면 신청",
                       "교차전부", "교차 전부", "편입생", "전부(과)", "학위수여식", "학사학위취득",
                       "오리엔테이션", "반편성고사", "합격자 공고", "합격자공고", "합격자 발표", "합격자발표",
                       "선발 결과", "선발결과", "이수 면제", "이수면제", "수업운영 안내", "출결",
                       "중간고사", "기말고사", "전공과목 변경", "전공변경", "다전공 신청",
                       "학석사연계", "합격자 공지", "합격자공지",
                       "학사경고", "자기설계전공", "교양필수", "상상력이노베이터",
                       # 시설·IT 행정
                       "TOPIK", "한국어능력시험",
                       "학술정보관",
                       "예방접종",
                       "정전",
                       "소방 시설", "소방시설",
                       "교체공사", "보수 공사", "공사 안내",
                       "우편취급국", "국제우편",
                       "전산서비스", "종합정보시스템", "유무선인터넷", "유.무선",
                       # 인사·학교법인 행정
                       "퇴직교원",
                       "노동조합",
                       "개방이사",
                       # 학교기업·캠퍼스 서비스
                       "케이키친",
                       "헤이영",
                       "스쿨버스",
                       "영업시간",
                       # 학생 활동 지침·설문
                       "학생단체",
                       "대학발전계획",
                       "중대본",
                       "코딩 라운지"],
    "창업":           ["창업", "창업동아리", "창업지원", "창업멘토링", "스타트업", "아이디어톤",
                       "입주기업", "예비창업", "학생 CEO", "CEO 발굴",
                       "메이커"],
    "국제교류":       ["교환학생", "어학연수", "파견", "글로벌버디", "국제교류", "해외", "어학", "글로컬",
                       "글로벌 튜터링", "글로벌 Conversation", "글로벌 컨버세이션", "글로벌 문화 소통",
                       "단기연수", "단기 연수", "K-Move", "WEST 연수", "한·미대학생",
                       "글로벌 동행", "글로벌동행"],
    "교육/특강":      ["특강", "교육생", "아카데미", "KDT", "K-디지털", "강좌", "교육과정", "역량강화",
                       "평생교육", "RISE", "마이크로디그리", "TOPCIT", "연구방법론",
                       "초청강연", "특별강연", "핵심역량진단", "HS-CESA", "K-CESA", "UICA", "K-NSSE",
                       "폭력예방교육", "필수교육", "전문과정", "SW마에스트로", "코딩 캠프", "코딩캠프",
                       "신규 교과목", "재정데이터", "직업흥미검사", "심리증진",
                       "연구윤리", "워크숍", "진로지도시스템", "진로 캠프", "진로캠프",
                       "심폐소생술", "저작권", "청년인생설계", "기초역량 가이드", "미디어 클래스",
                       "과학살롱", "기초학문",
                       "고시반", "인문학 캠프", "블록체인 밋업", "진로탐색", "GPU 활용"],
    "공모전/경진대회": ["공모전", "경진대회", "챌린지", "해커톤", "대회", "공모", "문학상"],
    "봉사/서포터즈":  ["서포터즈", "서포터스", "봉사", "멘토", "봉사자", "기자단", "자원활동", "멘토단",
                       "멘토링", "자원봉사", "홍보대사", "하랑", "소통-e", "앰버서더",
                       "방송국 HBS", "홍보단", "수습기자", "운영자문위원", "모니터링단", "자문단",
                       "바로알림단", "기획단", "체험단", "발굴단", "순찰대", "제작단",
                       "자원지도자", "볼런톤", "청백리포터", "Friends of Korea",
                       "안전요원", "포모플로깅", "넷제로프렌즈", "KEPCO프렌즈"],
}


def infer_category(title: str) -> str:
    for prefix, cat in CATEGORY_PREFIX.items():
        if title.startswith(prefix):
            return cat
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in title for kw in keywords):
            return cat
    return "기타"


# ── 목록 페이지 파싱 ──────────────────────────────────────────

def fetch_page(page: int) -> list[dict]:
    """목록 페이지 한 장 → [{title, url, date}] 반환. 빈 페이지면 []."""
    try:
        res = requests.get(LIST_URL, params={"page": page}, headers=HEADERS, timeout=10)
        res.raise_for_status()
    except Exception as e:
        print(f"  ⚠️ 페이지 {page} 요청 실패: {e}")
        return []

    soup  = BeautifulSoup(res.text, "html.parser")
    rows  = soup.select("table.board-table tbody tr")
    items = []

    for row in rows:
        # 공지(고정) 행 건너뜀
        if row.select_one(".notice"):
            continue

        title_td = row.select_one("td.title a")
        date_td  = row.select_one("td.date")
        if not title_td or not date_td:
            continue

        title = title_td.get_text(strip=True)
        date  = date_td.get_text(strip=True)
        href  = title_td.get("href", "")
        url   = BASE_URL + href if href.startswith("/") else href

        items.append({"title": title, "url": url, "date": date})

    return items


# ── 메인 ─────────────────────────────────────────────────────

def main():
    # 기존 파일에서 처리된 URL 로드 (중복 방지)
    existing: list[dict] = []
    seen_urls: set[str]  = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        seen_urls = {item["url"] for item in existing}
        print(f"기존 {len(existing)}건 로드 — 중복 건너뜀")

    results   = list(existing)
    page      = 1
    new_count = 0
    stop      = False

    print(f"{YEAR}년 공지 제목 수집 시작...\n")

    while not stop:
        items = fetch_page(page)
        if not items:
            print(f"  페이지 {page}: 결과 없음 — 종료")
            break

        for item in items:
            year = item["date"][:4]

            if year > YEAR:
                # 더 최신 연도 — 건너뜀
                continue

            if year < YEAR:
                # 2025년 이전 데이터 도달 — 전체 중단
                stop = True
                break

            # year == YEAR
            if item["url"] in seen_urls:
                continue

            item["category"] = infer_category(item["title"])
            results.append(item)
            seen_urls.add(item["url"])
            new_count += 1
            print(f"  [{new_count:4d}] [{item['category']}] {item['title'][:60]}")

        if not stop:
            page += 1
            time.sleep(DELAY)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {OUT_PATH} (신규 {new_count}건 추가 / 총 {len(results)}건)")


if __name__ == "__main__":
    main()
