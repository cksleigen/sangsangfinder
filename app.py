# ============================================================
# app.py — 상상파인더 (온보딩 → 사이드바 + 챗봇/추천게시물)
# ============================================================

import os, re, time, json, warnings, logging
# suppress chromadb 0.6.3 telemetry bug noise before any chromadb import
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
from crawling.crawler import get_post_content  # noqa: E402  (중복 구현 제거)
from api.core.config import (
    CHROMA_DB_PATH, NOTICES_CACHE_PATH,
    BOARD_LIST_URL, HEADERS, TARGET_YEAR, CATEGORIES, CATEGORY_PREFIX, CATEGORY_KEYWORDS,
    CATEGORY_PATTERN as _CATEGORY_PATTERN, SUFFIX_PATTERN as _SUFFIX_PATTERN,
)
from api.core.models import (
    get_embed_model, get_summary_pipeline, get_classifier, get_chroma,
    classify_notice, load_notices_cache, index_notices,
)
from api.services.search_service import hybrid_search, generate_llm_reply, invalidate_bm25_cache
from api.services.recommend_service import recommend_notices, summarize_notice

import streamlit as st

warnings.filterwarnings("ignore")

# ── 경로 설정 (프로젝트 루트 기준 상대경로) ───────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROFILE_CACHE_PATH = os.path.join(_BASE_DIR, "data", "profile_cache.json")

os.makedirs(os.path.join(_BASE_DIR, "data"), exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# 단과대 / 트랙·학과 매핑
COLLEGE_MAP = {
    "크리에이티브인문예술대학": ["영미문화콘텐츠트랙", "영미언어정보트랙", "한국어교육트랙", "역사문화큐레이션트랙", "역사콘텐츠트랙", "지식정보문화트랙", "디지털인문정보학트랙", "동양화전공", "서양화전공", "한국무용전공", "현대무용전공", "발레전공"],
    "미래융합사회과학대학": ["국제무역트랙", "글로벌비지니스트랙", "기업ㆍ경제분석트랙", "경제금융투자트랙", "공공행정트랙", "법&정책트랙", "부동산트랙", "스마트도시ㆍ교통계획트랙", "기업경영트랙", "비지니스애널리틱스트랙", "회계ㆍ재무경영트랙"],
    "디자인대학": ["패션마케팅트랙", "패션디자인트랙", "패션크리에이티브디렉션트랙", "미디어디자인트랙", "시각디자인트랙", "영상ㆍ애니메이션디자인트랙", "UX/UI디자인트랙", "인테리어디자인트랙", "VMDㆍ전시디자인트랙", "게임그래픽디자인트랙", "뷰티디자인매니지먼트학과"],
    "IT공과대학": ["모바일소프트웨어트랙", "빅데이터트랙", "디지털콘텐츠ㆍ가상현실트랙", "웹공학트랙", "전자트랙", "시스템반도체트랙", "기계시스템디자인트랙", "AI로봇융합트랙", "산업공학트랙", "응용산업데이터공학트랙"],
    "창의융합대학": ["상상력인재학부", "문학문화콘텐츠학과", "AI응용학과", "융합보안학과", "미래모빌리티학과"],
    "글로벌인재대학": ["한국언어문화교육학과", "글로벌K비지니스학과", "영상엔터테인먼트학과", "패션뷰티크리에이션학과", "SW융합학과", "글로벌벤처창업학과"],
    "미래플러스대학": ["융합행정학과", "호텔외식경영학과", "뷰티디자인학과", "비지니스컨설팅학과", "ICT융합디자인학과", "AIㆍ소프트웨어학과", "뷰티매니지먼트학과", "디지털콘텐츠디자인학과", "인테리어디자인학과", "스마트제조혁신컨설팅학과"],
}



# ============================================================
# 로고 이미지 로드
# ============================================================

def _load_image_b64(filename: str) -> str:
    import base64
    path = os.path.join(_BASE_DIR, filename)
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def get_logo_base64() -> str:
    return _load_image_b64("logo.png")


def get_hsu_base64() -> str:
    return _load_image_b64("hsu.png")

# ============================================================
# 유틸
# ============================================================

def clean_url(url: str) -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    params.pop("layout", None)
    new_query = urlencode({k: v[0] for k, v in params.items()})
    return urlunparse(parsed._replace(query=new_query))


def clean_title(raw: str) -> str:
    title = raw.replace("\n", " ").replace("\r", " ")
    title = re.sub(r"\s{2,}", " ", title).strip()
    title = _CATEGORY_PATTERN.sub("", title).strip()
    title = _SUFFIX_PATTERN.sub("", title).strip()
    return title




# ============================================================
# 크롤러
# ============================================================
# _ocr_image, _extract_pdf, get_post_content → crawler.py 참고


def get_list_page(page: int):
    try:
        res = requests.get(BOARD_LIST_URL, params={"page": page},
                           headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup  = BeautifulSoup(res.text, "html.parser")
        items = []
        for tr in soup.find_all("tr"):
            if not tr.find_all("td"):
                continue
            tr_classes = tr.get("class") or []
            if "notice" in tr_classes:
                continue
            date_el = tr.select_one(".td-date")
            link_el = tr.select_one(".td-title a")
            if not date_el or not link_el:
                continue
            date_text = date_el.get_text(strip=True)
            if not date_text.startswith(TARGET_YEAR):
                if page >= 5:
                    return items, True
                else:
                    continue
            href = link_el.get("href", "")
            if href.startswith("/"):
                href = "https://www.hansung.ac.kr" + href
            items.append({
                "title": clean_title(link_el.get_text()),
                "url":   clean_url(href),
                "date":  date_text,
            })
        return items, False
    except Exception as e:
        print(f"  ⚠️ 목록 파싱 실패 (page={page}): {e}")
        return [], False


def crawl_all() -> list:
    all_items, page = [], 1
    while True:
        items, done = get_list_page(page)
        if items:
            all_items.extend(items)
        if done or not items:
            break
        page += 1
        time.sleep(0.3)
    for item in all_items:
        item["body"]     = get_post_content(item["url"])
        item["category"] = classify_notice(item["title"], item["body"])
        time.sleep(0.2)
    with open(NOTICES_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    return all_items




# ============================================================
# CSS
# ============================================================

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #f2f2f7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Noto Sans KR", sans-serif !important;
}
/* 메인 컨텐츠 여백 축소 */
.block-container {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 860px !important;
    margin: 0 auto !important;
}

/* 사이드바 */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
.st-emotion-cache-h5rgaw,
[data-testid="stSidebar"] > div:first-child > div:first-child button { display: none !important; }
section[data-testid="stSidebar"] {
    width: 260px !important;
    min-width: 260px !important;
    transform: translateX(0) !important;
    visibility: visible !important;
    background: #ffffff !important;
    border-right: 1px solid rgba(0,0,0,0.08) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    width: 260px !important;
    padding: 8px 16px 16px 16px !important;
}
[data-testid="stSidebar"] * {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Noto Sans KR", sans-serif !important;
}

/* 버튼 */
.stButton > button {
    background: #0a84ff !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 7px 14px !important;
    transition: all 0.15s ease !important;
    letter-spacing: -0.01em !important;
}
.stButton > button:hover {
    background: #409cff !important;
    transform: scale(1.01) !important;
}

/* 인풋 */
.stTextInput > div > div > input {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0a84ff !important;
    box-shadow: 0 0 0 3px rgba(10,132,255,0.15) !important;
    outline: none !important;
}

/* 셀렉트박스 */
.stSelectbox > div > div {
    background: white !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
}

/* 멀티셀렉트 */
.stMultiSelect > div > div {
    background: white !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
}
/* 멀티셀렉트 체크박스 파란색 */
.stMultiSelect [data-baseweb="checkbox"] svg { color: #0a84ff !important; fill: #0a84ff !important; }
.stMultiSelect [aria-selected="true"] { background: rgba(10,132,255,0.08) !important; color: #0a84ff !important; }
/* 라디오 파란색 */
.stRadio [data-baseweb="radio"] div { border-color: #0a84ff !important; }
.stRadio [data-baseweb="radio"] [data-checked="true"] div { background: #0a84ff !important; border-color: #0a84ff !important; }
/* 전체 accent 빨강 → 파란색 override (config.toml 없이도 동작) */
:root {
    --primary-color: #0a84ff !important;
    --primary-hue: 211 !important;
}
/* 라디오 버튼 */
[data-baseweb="radio"] [data-checked="true"] > div:first-child {
    background-color: #0a84ff !important;
    border-color: #0a84ff !important;
}
[data-baseweb="radio"] > div:first-child {
    border-color: #0a84ff !important;
}
/* 멀티셀렉트 태그 */
[data-baseweb="tag"] { background-color: rgba(10,132,255,0.12) !important; }
[data-baseweb="tag"] span { color: #0a84ff !important; }
/* 체크박스 */
[data-baseweb="checkbox"] [data-checked="true"] > div {
    background-color: #0a84ff !important;
    border-color: #0a84ff !important;
}
/* 포커스 링 */
*:focus-visible { outline-color: #0a84ff !important; }
/* 탭 underline */
[data-baseweb="tab-highlight"] { background-color: #0a84ff !important; }
/* 프로그레스/스피너 */
[data-testid="stProgress"] > div > div { background-color: #0a84ff !important; }
/* 슬라이더 */
[data-baseweb="slider"] [role="slider"] { background-color: #0a84ff !important; border-color: #0a84ff !important; }
[data-baseweb="slider"] div[data-testid="stSliderTrackFill"] { background-color: #0a84ff !important; }

/* 라디오 */
.stRadio > div { gap: 4px !important; }

/* 탭 */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(120,120,128,0.12) !important;
    border-radius: 12px !important;
    padding: 3px !important;
    gap: 2px !important;
    width: fit-content !important;
    margin: 0 auto 16px auto !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #3c3c43 !important;
    padding: 8px 20px !important;
    font-family: -apple-system, "SF Pro Text", "Noto Sans KR", sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1d1d1f !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
    font-weight: 600 !important;
}
/* 탭 하단 액센트 라인 파스텔 블루로 */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #7ab8f5 !important;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: transparent !important;
}

/* 채팅 */
.chat-bubble-user {
    background: #0a84ff;
    color: white;
    border-radius: 20px 20px 5px 20px;
    padding: 10px 16px;
    max-width: 55%;
    width: fit-content;
    margin-left: auto;
    margin-bottom: 10px;
    font-size: 14px;
    line-height: 1.5;
    word-break: break-word;
}
.chat-bubble-bot {
    background: #f0f4ff;
    color: #1d1d1f;
    border-radius: 20px 20px 20px 5px;
    padding: 12px 18px;
    max-width: 75%;
    width: fit-content;
    margin-bottom: 10px;
    font-size: 14px;
    line-height: 1.55;
    border: 1px solid #dce8ff;
    word-break: break-word;
}
/* 맥OS 창 버튼 */
.mac-bar {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 10px 16px;
    background: #e8eef5;
    border-radius: 10px 10px 0 0;
    margin: -14px -14px 0 -14px;
    border-bottom: 1px solid rgba(0,0,0,0.07);
}
.mac-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}
.mac-dot-red    { background: #ff5f57; box-shadow: 0 0 0 0.5px rgba(0,0,0,0.12); }
.mac-dot-yellow { background: #febc2e; box-shadow: 0 0 0 0.5px rgba(0,0,0,0.12); }
.mac-dot-green  { background: #28c840; box-shadow: 0 0 0 0.5px rgba(0,0,0,0.12); }

/* mac-bar를 감싸는 Streamlit 요소 여백 제거 */
[data-testid="stMarkdownContainer"]:has(.mac-bar) {
    margin-bottom: -1rem !important;
    line-height: 0 !important;
}

/* 채팅 컨테이너 */
[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-bubble-user),
[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-bubble-bot),
[data-testid="stVerticalBlockBorderWrapper"]:has(.mac-bar) {
    background: #f9f9fb !important;
    border-radius: 13px !important;
    min-height: 0 !important;
    border: 1px solid rgba(0,0,0,0.07) !important;
    padding: 14px !important;
}

/* 빈 상태일 때도 동일하게 */
[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"]) + div {
    margin-top: 0 !important;
}

/* 공지 카드 */
.notice-card {
    background: white;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.05);
    transition: box-shadow 0.15s ease;
}
.notice-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); }
.notice-tag {
    display: inline-block;
    background: rgba(10,132,255,0.1);
    color: #0a84ff;
    border-radius: 7px;
    padding: 2px 9px;
    font-size: 11px;
    font-weight: 700;
    margin-right: 8px;
    letter-spacing: 0.02em;
}
.notice-title {
    font-size: 15px;
    font-weight: 600;
    color: #1d1d1f;
    margin: 6px 0 2px 0;
    line-height: 1.4;
}
.notice-date { font-size: 12px; color: #86868b; }
.notice-summary {
    font-size: 13px;
    color: #3c3c43;
    margin-top: 8px;
    line-height: 1.55;
}

/* 사이드바 섹션 레이블 */
.sb-label {
    font-size: 10px;
    font-weight: 700;
    color: rgba(0,0,0,0.35);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 18px 0 6px 2px;
}
.sb-info-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 5px 0;
}
.sb-info-key {
    font-size: 12px;
    color: rgba(0,0,0,0.4);
    min-width: 52px;
}
.sb-info-val {
    font-size: 13px;
    font-weight: 500;
    color: #1d1d1f;
    line-height: 1.4;
}

/* 온보딩 카드 — column 컨테이너에 직접 적용 */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: white !important;
    border-radius: 20px !important;
    padding: 32px 36px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08) !important;
    border: none !important;
}

/* 사이드바 스크롤 고정 */
section[data-testid="stSidebar"] > div:first-child {
    overflow: hidden !important;
    overflow-y: hidden !important;
}

/* 사이드바 최상단 강제 여백 제거 */
[data-testid="stSidebar"] > div:first-child > div:first-child {
    padding-top: 0 !important;
    margin-top: -3rem !important;
}

/* 구분선 */
hr { border: none; border-top: 1px solid rgba(0,0,0,0.08) !important; margin: 14px 0 !important; }

/* 숨기기 */
#MainMenu, footer, header { visibility: hidden; }
</style>
"""


# ============================================================
# 온보딩 화면
# ============================================================

def render_onboarding():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)

    # 왼쪽(로고/설명) + 오른쪽(폼) 2분할
    col_logo, col_form = st.columns([1, 2], gap="large")

    # ── 왼쪽: 로고 + 타이틀 ──────────────────────────────────
    with col_logo:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
        logo_b64 = get_logo_base64()
        logo_img = f'<img src="data:image/png;base64,{logo_b64}" style="width:90px; height:auto; object-fit:contain; display:block;">' \
                   if logo_b64 else '<div style="font-size:56px;">🔍</div>'
        st.markdown(f"""
<div style="padding-left:8px;">
  {logo_img}
  <div style="font-size:28px; font-weight:700; color:#1d1d1f; letter-spacing:-0.03em; margin-top:16px;">상상파인더</div>
  <div style="font-size:14px; color:#86868b; margin-top:6px; line-height:1.6;">
    한성대 공지를<br>스마트하게 검색하세요.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 오른쪽: 입력 폼 ──────────────────────────────────────
    with col_form:
        with st.container(border=True):
            st.markdown("""
<div style="font-size:17px; font-weight:700; color:#1d1d1f; margin-bottom:3px;">반갑습니다 👋</div>
<div style="font-size:13px; color:#86868b; margin-bottom:14px;">기본 정보를 알려주세요.</div>
""", unsafe_allow_html=True)

            # 1행: 이름 + 단과대
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                name = st.text_input("이름", placeholder="홍길동", key="ob_name")
            with r1c2:
                college = st.selectbox("단과대", list(COLLEGE_MAP.keys()), key="ob_college")

            # 2행: 트랙/학과 + 학년
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                track_options = COLLEGE_MAP.get(college, ["기타"])
                track = st.selectbox("트랙 / 학과", track_options, key="ob_track")
            with r2c2:
                grade = st.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"], key="ob_grade")

            # 3행: 관심사 (한 줄 전체)
            interests = st.multiselect(
                "관심사",
                CATEGORIES + ["교환학생"],
                placeholder="관심 카테고리를 선택하세요",
                key="ob_interests"
            )

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            # 4행: 버튼
            if st.button("시작하기 →", use_container_width=True):
                if not name.strip():
                    st.markdown('<div style="background:#e8f1ff;color:#0a84ff;border-radius:10px;padding:10px 16px;font-size:14px;font-weight:500;border:1px solid #b3d1ff;">✏️ 이름을 입력해 주세요.</div>', unsafe_allow_html=True)
                else:
                    profile_data = {
                        "name": name.strip(),
                        "college": college,
                        "track": track,
                        "grade": grade,
                        "interests": interests,
                    }
                    st.session_state.profile   = profile_data
                    st.session_state.onboarded = True
                    with open(PROFILE_CACHE_PATH, "w", encoding="utf-8") as f:
                        json.dump(profile_data, f, ensure_ascii=False, indent=2)
                    st.rerun()


# ============================================================
# 사이드바
# ============================================================

def render_sidebar(profile: dict):
    with st.sidebar:
        # 로고
        logo_b64 = get_logo_base64()
        logo_img = f'<img src="data:image/png;base64,{logo_b64}" style="width:32px; height:32px; object-fit:contain; flex-shrink:0;">' if logo_b64 else '<div style="width:32px; height:32px; font-size:18px; display:flex; align-items:center; justify-content:center;">🔍</div>'
        st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px; padding-bottom:10px;">
  {logo_img}
  <div>
    <div style="font-size:14px; font-weight:700; color:#1d1d1f; letter-spacing:-0.02em;">상상파인더</div>
    <div style="font-size:10px; color:#86868b; margin-top:1px;">Hansung Notice Finder</div>
  </div>
</div>
<hr/>
""", unsafe_allow_html=True)

        # 사용자 정보
        st.markdown('<div class="sb-label">내 정보</div>', unsafe_allow_html=True)

        rows = [
            ("이름", profile.get("name", "")),
            ("단과대학", profile.get("college", "")),
            ("트랙/학과", profile.get("track", "")),
            ("학년", profile.get("grade", "")),
        ]
        html = ""
        for key, val in rows:
            html += f"""
<div class="sb-info-row">
  <span class="sb-info-key">{key}</span>
  <span class="sb-info-val">{val}</span>
</div>"""
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # 바로가기
        st.markdown('<div class="sb-label">바로가기</div>', unsafe_allow_html=True)

        # 트랙/학과별 홈페이지 URL 매핑
        DEPT_URLS = {
            # 글로벌인재대학
            "한국언어문화교육학과":      "https://www.hansung.ac.kr/global/1511/subview.do",
            "글로벌K비지니스학과":       "https://www.hansung.ac.kr/global/1516/subview.do",
            "영상엔터테인먼트학과":      "https://www.hansung.ac.kr/global/1521/subview.do",
            "패션뷰티크리에이션학과":     "https://www.hansung.ac.kr/global/1526/subview.do",
            "SW융합학과":             "https://www.hansung.ac.kr/global/1531/subview.do",
            "글로벌벤처창업학과":        "https://www.hansung.ac.kr/global/6807/subview.do",
            # 미래플러스대학
            "융합행정학과":            "https://www.hansung.ac.kr/futureplus/731/subview.do",
            "호텔외식경영학과":          "https://www.hansung.ac.kr/futureplus/734/subview.do",
            "뷰티디자인학과":           "https://www.hansung.ac.kr/futureplus/737/subview.do",
            "비지니스컨설팅학과":        "https://www.hansung.ac.kr/futureplus/740/subview.do",
            "ICT융합디자인학과":        "https://www.hansung.ac.kr/futureplus/743/subview.do",
            "AIㆍ소프트웨어학과":       "https://www.hansung.ac.kr/futureplus/746/subview.do",
            "뷰티매니지먼트학과":        "https://www.hansung.ac.kr/futureplus/749/subview.do",
            "디지털콘텐츠디자인학과":     "https://www.hansung.ac.kr/futureplus/754/subview.do",
            "인테리어디자인학과":        "https://www.hansung.ac.kr/futureplus/759/subview.do",
            "스마트제조혁신컨설팅학과":   "https://www.hansung.ac.kr/futureplus/764/subview.do",
            # 창의융합대학
            "상상력인재학부":    "https://www.hansung.ac.kr/CreCon/2761/subview.do",
            "문학문화콘텐츠학과": "https://www.hansung.ac.kr/CreCon/2768/subview.do",
            "AI응용학과":       "https://www.hansung.ac.kr/CreCon/2777/subview.do",
            "융합보안학과":      "https://www.hansung.ac.kr/CreCon/2787/subview.do",
            "미래모빌리티학과":   "https://www.hansung.ac.kr/CreCon/2796/subview.do",
            # 미래융합사회과학대학
            "국제무역트랙":              "https://www.hansung.ac.kr/SclScn/5260/subview.do",
            "글로벌비지니스트랙":         "https://www.hansung.ac.kr/SclScn/5267/subview.do",
            "기업ㆍ경제분석트랙":         "https://www.hansung.ac.kr/SclScn/5274/subview.do",
            "경제금융투자트랙":           "https://www.hansung.ac.kr/SclScn/5281/subview.do",
            "공공행정트랙":              "https://www.hansung.ac.kr/SclScn/5295/subview.do",
            "법&정책트랙":              "https://www.hansung.ac.kr/SclScn/5303/subview.do",
            "부동산트랙":               "https://www.hansung.ac.kr/SclScn/5313/subview.do",
            "스마트도시ㆍ교통계획트랙":    "https://www.hansung.ac.kr/SclScn/5321/subview.do",
            "기업경영트랙":              "https://www.hansung.ac.kr/SclScn/5328/subview.do",
            "비지니스애널리틱스트랙":      "https://www.hansung.ac.kr/SclScn/5336/subview.do",
            "회계ㆍ재무경영트랙":         "https://www.hansung.ac.kr/SclScn/5344/subview.do",
            # IT공과대학
            "모바일소프트웨어트랙":        "https://www.hansung.ac.kr/Engineering/4887/subview.do",
            "빅데이터트랙":              "https://www.hansung.ac.kr/Engineering/4894/subview.do",
            "디지털콘텐츠ㆍ가상현실트랙":   "https://www.hansung.ac.kr/Engineering/4901/subview.do",
            "웹공학트랙":               "https://www.hansung.ac.kr/Engineering/4908/subview.do",
            "전자트랙":                 "https://www.hansung.ac.kr/Engineering/4915/subview.do",
            "시스템반도체트랙":           "https://www.hansung.ac.kr/Engineering/4922/subview.do",
            "기계시스템디자인트랙":        "https://www.hansung.ac.kr/Engineering/4929/subview.do",
            "AI로봇융합트랙":            "https://www.hansung.ac.kr/Engineering/4936/subview.do",
            "산업공학트랙":              "https://www.hansung.ac.kr/Engineering/4992/subview.do",
            "응용산업데이터공학트랙":       "https://www.hansung.ac.kr/Engineering/5020/subview.do",
            # 디자인대학
            "패션마케팅트랙":              "https://www.hansung.ac.kr/Design/5103/subview.do",
            "패션디자인트랙":              "https://www.hansung.ac.kr/Design/5110/subview.do",
            "패션크리에이티브디렉션트랙":    "https://www.hansung.ac.kr/Design/5117/subview.do",
            "미디어디자인트랙":            "https://www.hansung.ac.kr/Design/5124/subview.do",
            "시각디자인트랙":              "https://www.hansung.ac.kr/Design/5145/subview.do",
            "영상ㆍ애니메이션디자인트랙":    "https://www.hansung.ac.kr/Design/5131/subview.do",
            "UX/UI디자인트랙":            "https://www.hansung.ac.kr/Design/5173/subview.do",
            "인테리어디자인트랙":           "https://www.hansung.ac.kr/Design/5159/subview.do",
            "VMDㆍ전시디자인트랙":         "https://www.hansung.ac.kr/Design/5152/subview.do",
            "게임그래픽디자인트랙":         "https://www.hansung.ac.kr/Design/5166/subview.do",
            "뷰티디자인매니지먼트학과":      "https://www.hansung.ac.kr/Design/5180/subview.do",
            "영미문화콘텐츠트랙":         "https://www.hansung.ac.kr/HmnArt/5641/subview.do",
            "영미언어정보트랙":           "https://www.hansung.ac.kr/HmnArt/5577/subview.do",
            "한국어교육트랙":            "https://www.hansung.ac.kr/HmnArt/5584/subview.do",
            "역사문화큐레이션트랙":        "https://www.hansung.ac.kr/HmnArt/5627/subview.do",
            "역사콘텐츠트랙":            "https://www.hansung.ac.kr/HmnArt/5634/subview.do",
            "지식정보문화트랙":           "https://www.hansung.ac.kr/HmnArt/5613/subview.do",
            "디지털인문정보학트랙":        "https://www.hansung.ac.kr/HmnArt/5620/subview.do",
            "동양화전공":               "https://www.hansung.ac.kr/HmnArt/5648/subview.do",
            "서양화전공":               "https://www.hansung.ac.kr/HmnArt/5655/subview.do",
            "한국무용전공":              "https://www.hansung.ac.kr/HmnArt/5662/subview.do",
            "현대무용전공":              "https://www.hansung.ac.kr/HmnArt/5669/subview.do",
            "발레전공":                "https://www.hansung.ac.kr/HmnArt/5676/subview.do",
        }

        links = [
            ("🏫", "한성대학교", "https://www.hansung.ac.kr/hansung/index.do"),
            ("💻", "한성 e-class", "https://learn.hansung.ac.kr/"),
            ("📋", "종합정보시스템", "https://info.hansung.ac.kr/"),
            ("📊", "스마트자기관리시스템", "https://hsportal.hansung.ac.kr/"),
            ("📚", "학술정보관", "https://hsel.hansung.ac.kr/"),
        ]

        # 사용자 트랙/학과 홈페이지 추가
        track = profile.get("track", "")
        if track in DEPT_URLS:
            links.append(("🎓", f"{track}", DEPT_URLS[track]))
        link_html = ""
        for icon, label, url in links:
            link_html += f"""
<a href="{url}" target="_blank" style="
    display:flex; align-items:center; gap:9px;
    padding:8px 10px; border-radius:9px;
    text-decoration:none; color:#1d1d1f;
    font-size:13px; font-weight:500;
    transition:background 0.15s;
    margin-bottom:2px;
" onmouseover="this.style.background='rgba(0,0,0,0.05)'"
  onmouseout="this.style.background='transparent'">
  <span style="font-size:15px;">{icon}</span>
  <span>{label}</span>
  <span style="margin-left:auto; font-size:11px; color:#aeaeb2;">↗</span>
</a>"""
        st.markdown(link_html, unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("""
<style>
section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] .stButton button {
    font-size: 11px !important;
    padding: 4px 8px !important;
    height: 28px !important;
    min-height: 0 !important;
    line-height: 1 !important;
    border-radius: 6px !important;
}
/* 정보 다시 입력 버튼 중앙 정렬 */
section[data-testid="stSidebar"] .reset-btn {
    display: flex;
    justify-content: center;
}
section[data-testid="stSidebar"] .reset-btn button {
    width: auto !important;
}
</style>
<div class="reset-btn">
""", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([0.3, 2.4, 0.3])
        with btn_col:
            if st.button("내 정보 다시 입력", use_container_width=True):
                st.session_state.onboarded = False
                st.session_state.profile = {}
                st.session_state.chat_history = []
                if os.path.exists(PROFILE_CACHE_PATH):
                    os.remove(PROFILE_CACHE_PATH)
                st.rerun()






# ============================================================
# 메인 화면 — 챗봇
# ============================================================

def render_chatbot(profile: dict):
    top_k = 5
    alpha = 0.7

    # 채팅 영역 — mac 버튼 바 + 컨테이너를 하나로
    with st.container(border=True):
        # 맥OS 창 버튼 바 (컨테이너 안 최상단)
        st.markdown("""
<div class="mac-bar">
  <div class="mac-dot mac-dot-red"></div>
  <div class="mac-dot mac-dot-yellow"></div>
  <div class="mac-dot mac-dot-green"></div>
</div>
<div style="margin-top:12px;"></div>
""", unsafe_allow_html=True)
        if not st.session_state.chat_history:
            name = profile.get("name", "")
            hsu_b64 = get_hsu_base64()
            hsu_img = f'<img src="data:image/png;base64,{hsu_b64}" style="width:60px; height:60px; object-fit:contain; display:block; margin:0 auto 14px auto;">' if hsu_b64 else '<div style="font-size:38px; margin-bottom:14px;">🔍</div>'
            st.markdown(f"""
<div style="text-align:center; padding:24px 0 20px; color:#86868b;">
  {hsu_img}
  <div style="font-size:17px; font-weight:600; color:#1d1d1f; margin-bottom:6px;">
    안녕하세요, {name}님!
  </div>
  <div style="font-size:13px; line-height:1.6;">
    한성대 공지를 자연어로 검색해보세요.<br/>
    <span style="color:#aeaeb2;">"장학금 신청 기간 알려줘" &nbsp;·&nbsp; "취업박람회 언제야?" &nbsp;·&nbsp; "비교과 프로그램 추천해줘"</span>
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-bubble-user">{msg["content"]}</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="chat-bubble-bot">{msg["content"]}</div>',
                        unsafe_allow_html=True)
                    if msg.get("results"):
                        r = msg["results"][0]
                        st.markdown(f"""
<div class="notice-card">
  <span style="font-size:11px;color:#86868b;font-weight:500;">📎 참고 공지</span>
  &nbsp;<span class="notice-tag">{r.get('category','기타')}</span>
  <span class="notice-date">{r['date']}</span>
  <div class="notice-title">{r['title']}</div>
  <div style="margin-top:8px;">
    <a href="{r['url']}" target="_blank"
       style="font-size:12px;color:#0a84ff;text-decoration:none;font-weight:600;">
      공지 바로가기 →
    </a>
  </div>
</div>
""", unsafe_allow_html=True)

    # 입력 폼: 카테고리 | 입력창 | 전송
    with st.form("chat_form", clear_on_submit=True):
        c0, c1, c2 = st.columns([1.2, 4, 0.8])
        with c0:
            cat_filter = st.selectbox("카테고리", ["전체"] + CATEGORIES,
                                      label_visibility="collapsed", key="chat_cat")
        with c1:
            user_input = st.text_input(
                "메시지", placeholder="무엇이 궁금하세요?",
                label_visibility="collapsed")
        with c2:
            submitted = st.form_submit_button("전송", use_container_width=True)

    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        cat_filter = st.session_state.get("chat_cat", "전체")
        results = hybrid_search(
            user_input, top_k=top_k, alpha=alpha,
            category_filter=cat_filter if cat_filter != "전체" else None,
        )
        with st.spinner("답변 생성 중..."):
            is_first = len(st.session_state.chat_history) == 1  # user 메시지 추가된 직후
            reply = generate_llm_reply(user_input, results, st.session_state.profile, is_first=is_first)
        st.session_state.chat_history.append(
            {"role": "bot", "content": reply, "results": results})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("대화 초기화", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================
# 메인 화면 — 추천 게시물
# ============================================================

def render_recommend(profile: dict):
    st.markdown(f"""
<div style="background:white; border-radius:14px; padding:16px 20px; margin-bottom:20px;
     box-shadow:0 1px 5px rgba(0,0,0,0.06); display:flex; align-items:center; gap:16px;">
  <div style="font-size:30px;">🎓</div>
  <div>
    <div style="font-size:15px; font-weight:700; color:#1d1d1f;">
      {profile.get('college','')} &nbsp;·&nbsp; {profile.get('track','')} &nbsp;·&nbsp; {profile.get('grade','')}
    </div>
    <div style="font-size:13px; color:#86868b; margin-top:3px;">
      관심사: {', '.join(profile.get('interests',[])) or '없음'}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([2, 1.5, 2])
    with col_c:
        btn_rec = st.button("맞춤 공지 추천받기", type="primary", use_container_width=True)
    if btn_rec:
        with st.spinner("추천 중..."):
            recs = recommend_notices(profile, top_k=5)
        if not recs:
            st.info("추천 결과가 없습니다. 먼저 캐시를 불러와 주세요.")
        else:
            body_map = {n["url"]: n.get("body", "")
                        for n in (st.session_state.notices or load_notices_cache())}
            for r in recs:
                body    = body_map.get(r["url"], "")
                summary = summarize_notice(r["title"], body) if body else ""
                st.markdown(f"""
<div class="notice-card">
  <span class="notice-tag">{r.get('category','기타')}</span>
  <span class="notice-date">{r['date']}</span>
  <div class="notice-title">{r['title']}</div>
  {"<div class='notice-summary'>" + summary + "</div>" if summary else ""}
  <div style="margin-top:10px;">
    <a href="{r['url']}" target="_blank"
       style="font-size:12px;color:#0a84ff;text-decoration:none;font-weight:600;">
      공지 바로가기 →
    </a>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 엔트리포인트
# ============================================================

def main():
    from PIL import Image
    try:
        hsu_icon = Image.open(os.path.join(_BASE_DIR, "hsu.png"))
    except Exception:
        hsu_icon = "🔍"
    st.set_page_config(
        page_title="상상파인더",
        page_icon=hsu_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 세션 초기화
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "notices"      not in st.session_state: st.session_state.notices      = []
    if "onboarded" not in st.session_state:
        # 저장된 프로필 있으면 자동 로드
        if os.path.exists(PROFILE_CACHE_PATH):
            with open(PROFILE_CACHE_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            if saved:
                st.session_state.profile  = saved
                st.session_state.onboarded = True
            else:
                st.session_state.profile   = {}
                st.session_state.onboarded = False
        else:
            st.session_state.profile   = {}
            st.session_state.onboarded = False

    # ── 온보딩 전 ─────────────────────────────────────────────
    if not st.session_state.onboarded:
        render_onboarding()
        return

    # ── 온보딩 후 ─────────────────────────────────────────────
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    profile = st.session_state.profile

    # 캐시 자동 로드 (최초 1회)
    if not st.session_state.notices:
        notices = load_notices_cache()
        if notices:
            st.session_state.notices = notices
            # ChromaDB에 이미 데이터가 있으면 index_notices 생략 (속도 최적화)
            if get_chroma().count() == 0:
                index_notices(notices)
                invalidate_bm25_cache()

    render_sidebar(profile)

    # 메인 탭
    tab_chat, tab_rec = st.tabs(["  💬 챗봇 검색  ", "  ✨ 추천 게시물  "])

    with tab_chat:
        render_chatbot(profile)

    with tab_rec:
        render_recommend(profile)


if __name__ == "__main__":
    main()
