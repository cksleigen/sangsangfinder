# ============================================================
# recommend.py — Two-Tower 추천 시스템 로직
# ============================================================

import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from sentence_transformers import SentenceTransformer
from supabase import create_client

import streamlit as st

# ── 경로 설정 ─────────────────────────────────────────────────
_BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
TWO_TOWER_MODEL_PATH = os.path.join(_BASE_DIR, "models", "two_tower_model.pt")
BASE_MODEL_EMBED     = "jhgan/ko-sroberta-multitask"
SUPABASE_URL         = os.getenv("SUPABASE_URL", "your_supabase_url")
SUPABASE_KEY         = os.getenv("SUPABASE_KEY", "your_supabase_key")

# ── 추천 가중치 ───────────────────────────────────────────────
MODEL_WEIGHT    = 0.4
CATEGORY_WEIGHT = 0.3
JOB_TYPE_WEIGHT = 0.2
SCORE_WEIGHT    = 0.1

PENALTY_CATEGORIES     = ['국제교류', '봉사/서포터즈', '창업']
NO_JOB_TYPE_CATEGORIES = ['국제교류', '장학금', '학사행정', 'ROTC', '기숙사/생활관', '봉사/서포터즈']

# ── job_type 정의 ─────────────────────────────────────────────
JOB_TYPE_SENTENCES = {
    "취업/채용": {
        "IT/개발":       "소프트웨어 개발 프로그래밍 AI 데이터 ICT 시스템 엔지니어 백엔드 프론트엔드 웹 앱 반도체 IT",
        "경영/금융":     "경영 회계 재무 마케팅 영업 금융 은행 증권 보험 캐피탈 투자 무역 물류",
        "디자인/마케팅": "디자인 마케팅 브랜드 콘텐츠 뷰티 패션 크리에이티브 광고 홍보 영상",
        "공공/연구":     "공공기관 연구원 정부 공무원 학술 R&D 재단 진흥원 공단 기관 연구소",
        "교내채용":      "조교 행정 한성대학교 교내 사업단 센터 계약직 임시직 학술연구원",
    },
    "비교과": {
        "IT교육":    "코딩 프로그래밍 소프트웨어 AI 데이터 파이썬 알고리즘 개발 SW",
        "진로/취업": "진로 취업 직무 커리어 멘토링 포트폴리오 자기소개서 면접 직업",
        "디자인":    "디자인 포토샵 영상편집 UX 그래픽 일러스트 미디어 콘텐츠",
        "글쓰기":    "글쓰기 에세이 작문 논문 독서 토론 스피치 발표",
        "심리/상담": "심리 상담 멘탈 스트레스 힐링 마음 정서 치유 감정",
        "ESG/봉사":  "ESG 환경 봉사 사회공헌 탄소 지속가능 서포터즈 동아리",
    },
    "교육/특강": {
        "AI/IT":     "AI 인공지능 TOPCIT 디지털 코딩 소프트웨어 빅데이터 블록체인",
        "진로/취업": "진로 취업 직무 커리어 면접 이력서 직업 멘토링",
        "어학":      "영어 어학 외국어 TOEIC 회화 글로벌 영문 언어",
        "디자인":    "디자인 UX UI 영상 포토샵 그래픽 미디어 콘텐츠",
        "인문/교양": "인문 교양 역사 철학 문화 예술 사회 경제 문학",
        "창업":      "창업 스타트업 아이디어 사업 기업가정신 벤처",
    },
    "공모전/경진대회": {
        "개발/SW":     "SW AI 소프트웨어 개발 코딩 알고리즘 해커톤 프로그래밍 IT",
        "디자인":      "디자인 UX UI 그래픽 영상 포스터 브랜딩 패션 시각 이미지",
        "창업/마케팅": "창업 아이디어 마케팅 비즈니스 스타트업 사업계획서 경영",
        "인문/사회":   "에세이 영어 프레젠테이션 영자신문 스피치 글쓰기 역사 문화",
    },
    "창업": {
        "IT창업":   "AI 기술 소프트웨어 플랫폼 테크 개발 서비스 앱",
        "소셜벤처": "소셜벤처 사회적기업 ESG 임팩트 환경 비영리 공익",
        "일반창업": "예비창업 CEO 아이디어 사업계획 창업지원 아카데미",
    },
    "국제교류": {
        "교환학생":   "교환학생 파견 해외대학 협정 학점인정 유학",
        "해외인턴":   "해외인턴 K-Move 청년봉사단 KOICA 해외현장실습 해외취업",
        "어학연수":   "어학연수 영어권 단기연수 어학캠프 언어",
        "외국인학생": "외국인 유학생 한국어 한국문화 글로벌학생 국제학생",
    },
}

TRACK_DOMAIN = {
    "IT": [
        "모바일소프트웨어트랙", "빅데이터트랙", "디지털콘텐츠ㆍ가상현실트랙",
        "웹공학트랙", "전자트랙", "시스템반도체트랙", "기계시스템디자인트랙",
        "AI로봇융합트랙", "산업공학트랙", "응용산업데이터공학트랙",
        "AI응용학과", "융합보안학과", "미래모빌리티학과",
        "SW융합학과", "글로벌벤처창업학과",
        "AIㆍ소프트웨어학과", "ICT융합디자인학과", "스마트제조혁신컨설팅학과",
    ],
    "경영": [
        "기업경영트랙", "회계ㆍ재무경영트랙", "경제금융투자트랙",
        "기업ㆍ경제분석트랙", "비지니스애널리틱스트랙",
        "국제무역트랙", "글로벌비지니스트랙",
        "글로벌K비지니스학과", "비지니스컨설팅학과", "호텔외식경영학과",
    ],
    "행정/공공": [
        "공공행정트랙", "법&정책트랙", "부동산트랙",
        "스마트도시ㆍ교통계획트랙", "융합행정학과",
    ],
    "디자인": [
        "패션마케팅트랙", "패션디자인트랙", "패션크리에이티브디렉션트랙",
        "미디어디자인트랙", "시각디자인트랙", "영상ㆍ애니메이션디자인트랙",
        "UX/UI디자인트랙", "인테리어디자인트랙", "VMDㆍ전시디자인트랙",
        "게임그래픽디자인트랙", "뷰티디자인매니지먼트학과",
        "패션뷰티크리에이션학과", "영상엔터테인먼트학과",
        "뷰티디자인학과", "뷰티매니지먼트학과",
        "디지털콘텐츠디자인학과", "인테리어디자인학과",
    ],
    "인문": [
        "영미문화콘텐츠트랙", "영미언어정보트랙", "한국어교육트랙",
        "역사문화큐레이션트랙", "역사콘텐츠트랙", "지식정보문화트랙",
        "디지털인문정보학트랙", "문학문화콘텐츠학과", "한국언어문화교육학과",
    ],
    "예술": [
        "동양화전공", "서양화전공",
        "한국무용전공", "현대무용전공", "발레전공",
    ],
    "융합": ["상상력인재학부"],
}

track_to_domain = {}
for _domain, _tracks in TRACK_DOMAIN.items():
    for _track in _tracks:
        track_to_domain[_track] = _domain
track_to_domain["트랙 미정"] = "융합"

DOMAIN_TO_JOB_TYPE = {
    "취업/채용":       {"IT": "IT/개발",  "경영": "경영/금융",  "행정/공공": "공공/연구",  "디자인": "디자인/마케팅", "인문": "공공/연구",  "예술": "디자인/마케팅", "융합": None},
    "비교과":          {"IT": "IT교육",   "경영": "진로/취업",  "행정/공공": "진로/취업",  "디자인": "디자인",        "인문": "글쓰기",     "예술": "디자인",        "융합": None},
    "교육/특강":       {"IT": "AI/IT",    "경영": "진로/취업",  "행정/공공": "인문/교양",  "디자인": "디자인",        "인문": "인문/교양",  "예술": "디자인",        "융합": None},
    "공모전/경진대회":  {"IT": "개발/SW",  "경영": "창업/마케팅","행정/공공": "인문/사회",  "디자인": "디자인",        "인문": "인문/사회",  "예술": "디자인",        "융합": None},
    "창업":            {"IT": "IT창업",   "경영": "일반창업",   "행정/공공": "소셜벤처",   "디자인": "일반창업",      "인문": "소셜벤처",   "예술": "소셜벤처",      "융합": None},
    "국제교류":        {"IT": "교환학생", "경영": "교환학생",   "행정/공공": "교환학생",   "디자인": "교환학생",      "인문": "어학연수",   "예술": "교환학생",      "융합": "교환학생"},
}

# ============================================================
# Supabase
# ============================================================

@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def calc_notice_score(notice: dict) -> float:
    score = 0.0
    score += min((notice.get('views', 0) or 0) / 1000, 0.4)
    try:
        date_str = str(notice.get('posted_date_text') or notice.get('posted_at', ''))[:10].replace('.', '-')
        days_ago = (datetime.now() - datetime.strptime(date_str, '%Y-%m-%d')).days
        if days_ago <= 7:    score += 0.4
        elif days_ago <= 30: score += 0.3
        elif days_ago <= 90: score += 0.15
    except Exception:
        pass
    if notice.get('category') in ['취업/채용', '장학금', '학사행정', '교육/특강']:
        score += 0.2
    return min(score, 1.0)

@st.cache_data(ttl=300, show_spinner=False)
def load_notices_from_supabase() -> list:
    try:
        supabase  = get_supabase()
        all_data  = []
        page_size = 1000
        offset    = 0
        while True:
            res   = supabase.table("notices").select(
                "id,notice_id,title,url,posted_at,posted_date_text,category,body,views,job_types"
            ).order("posted_at", desc=True).range(offset, offset + page_size - 1).execute()
            batch = res.data or []
            if not batch: break
            all_data.extend(batch)
            if len(batch) < page_size: break
            offset += page_size
        for n in all_data:
            n['notice_score'] = calc_notice_score(n)
            raw = str(n.get('posted_date_text') or n.get('posted_at', ''))
            raw = re.sub(r'<[^>]+>', '', raw).strip()
            n['date'] = raw[:10].replace('.', '-')
        if all_data:
            print(f"date 확인: {all_data[0]['date']}, raw: {all_data[0].get('posted_date_text')}")
        print(f"Supabase 공지 {len(all_data)}건 로드 완료")
        return all_data
    except Exception as e:
        print(f"Supabase 로드 오류: {e}")
        import traceback; traceback.print_exc()
        return []
    
@st.cache_data(ttl=300, show_spinner=False)
def load_embeddings_from_supabase() -> dict:
    try:
        supabase  = get_supabase()
        all_embs  = []
        page_size = 1000
        offset    = 0
        while True:
            res   = supabase.table("notices").select(
                "id,embedding"
            ).order("posted_at", desc=True).range(offset, offset + page_size - 1).execute()
            batch = res.data or []
            if not batch: break
            all_embs.extend(batch)
            if len(batch) < page_size: break
            offset += page_size

        def parse_embedding(emb):
            if emb is None:
                return np.zeros(128, dtype=np.float32)
            if isinstance(emb, str):
                emb = json.loads(emb)
            return np.array(emb, dtype=np.float32)

        # id → embedding 딕셔너리로 반환
        result = {r['id']: parse_embedding(r.get('embedding')) for r in all_embs}
        print(f"임베딩 {len(result)}건 로드 완료")
        return result
    except Exception as e:
        print(f"임베딩 로드 오류: {e}")
        return {}
    

# ============================================================
# Two-Tower 모델
# ============================================================

@st.cache_resource
def load_two_tower_model():
    device = torch.device('cpu')
    sbert  = SentenceTransformer(BASE_MODEL_EMBED, device="cpu")

    class UserTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128)
            )
        def forward(self, x):
            return F.normalize(self.fc(x), dim=-1)

    class ItemTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(769, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128)
            )
        def forward(self, x):
            return F.normalize(self.fc(x), dim=-1)

    class TwoTowerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_tower = UserTower()
            self.item_tower = ItemTower()
        def forward(self, u, i):
            return self.user_tower(u), self.item_tower(i)

    model = TwoTowerModel().to(device)
    if os.path.exists(TWO_TOWER_MODEL_PATH):
        model.load_state_dict(torch.load(TWO_TOWER_MODEL_PATH, map_location=device))
    model.eval()
    print("Two-Tower 모델 로드 완료!")
    return sbert, model, device

# ============================================================
# job_type 분류
# ============================================================

def classify_job_type(notice: dict, threshold: float = 0.35, top_k: int = 2) -> list:
    """Supabase에 미리 저장된 job_types 사용 (실시간 SBERT 호출 없음)"""
    job_types = notice.get('job_types', []) or []
    if isinstance(job_types, str):
        job_types = json.loads(job_types)
    return [{'job_type': jt, 'score': 1.0} for jt in job_types[:top_k]]

def get_user_domain(track: str) -> str:
    if "상상력인재학부" in track:
        for t, d in track_to_domain.items():
            if t in track and t != "상상력인재학부":
                return d
        return "융합"
    return track_to_domain.get(track, "융합")

def get_job_score(track: str, notice: dict) -> float:
    cat = notice.get('category', '')
    if cat in NO_JOB_TYPE_CATEGORIES:
        return 0.0
    notice_types = [t['job_type'] for t in classify_job_type(notice)]
    if not notice_types:
        return 0.0
    domain  = get_user_domain(track)
    user_jt = DOMAIN_TO_JOB_TYPE.get(cat, {}).get(domain)
    if user_jt is None:
        return 0.0
    return 1.0 if user_jt in notice_types else 0.0

# ============================================================
# Two-Tower 추천 (Supabase embedding 활용)
# ============================================================

def two_tower_recommend(college, track, year, interests, top_k=10):
    try:
        sbert, model, device = load_two_tower_model()
        notices              = load_notices_from_supabase()
        emb_dict             = load_embeddings_from_supabase()  # dict로 변경

        if not notices or not emb_dict:
            return []

        user_text   = f"{college} {track} {year} 관심사: {', '.join(interests)}"
        user_emb    = sbert.encode([user_text], convert_to_numpy=True)
        user_tensor = torch.tensor(user_emb, dtype=torch.float).to(device)
        with torch.no_grad():
            user_vec = model.user_tower(user_tensor).cpu().numpy()

        scores      = np.array([n.get('notice_score', 0) for n in notices])
        max_score   = scores.max() if scores.max() > 0 else 1
        scores_norm = scores / max_score

        results = []
        for n_idx, notice in enumerate(notices):
            nid     = notice.get('id')
            item_emb = emb_dict.get(nid)
            if item_emb is None:
                continue

            sim_score = float(np.dot(item_emb, user_vec.T).flatten()[0])
            category  = notice.get('category', '')
            n_score   = float(scores_norm[n_idx])
            sim_norm  = (sim_score + 1.0) / 2.0

            if category in interests:            cat_score = 1.0
            elif category in PENALTY_CATEGORIES: cat_score = -0.5
            else:                                cat_score = 0.0

            job_score = get_job_score(track, notice)

            final_score = (
                MODEL_WEIGHT    * sim_norm  +
                CATEGORY_WEIGHT * cat_score +
                JOB_TYPE_WEIGHT * job_score +
                SCORE_WEIGHT    * n_score
            )
            results.append({
                'notice':      notice,
                'final_score': final_score,
                'sim_score':   sim_norm,
                'cat_score':   cat_score,
                'job_score':   job_score,
            })

        results.sort(key=lambda x: x['final_score'], reverse=True)

        category_count = {}
        filtered = []
        for res in results:
            cat = res['notice'].get('category', '')
            if category_count.get(cat, 0) < 2:
                filtered.append(res)
                category_count[cat] = category_count.get(cat, 0) + 1
            if len(filtered) == top_k:
                break

        return filtered

    except Exception as e:
        print(f"Two-Tower 추천 오류: {e}")
        import traceback; traceback.print_exc()
        return []
