# Sangsangfinder

한성대학교 공지사항 검색 및 추천 시스템 — RAG(Retrieval-Augmented Generation) 파이프라인 기반

## 주요 기능

- 한성대 공지사항 RSS 크롤링 및 ChromaDB 적재 (주기적 크롤링: 우선 x)
- Hybrid Search: BM25(0.3) + Dense Vector(0.7) 결합 검색
- QA 생성-Judge 구조로 QA 데이터셋 자동 생성
- Streamlit UI를 통한 공지사항 검색 인터페이스

## 기술 스택

| 구분 | 내용 |
|------|------|
| 임베딩 모델 | `jhgan/ko-sroberta-multitask` |
| 검색 방식 | Hybrid (BM25 0.3 + Dense 0.7) |
| Vector DB | ChromaDB (Pinecone 마이그레이션 예정) |
| UI | Streamlit |
| QA 생성 | Claude API (Opus / Haiku) |
| 배포 | Streamlit Community Cloud (예정) |

## 디렉토리 구조

```
sangsangfinder/
├── app.py                      # Streamlit 검색 UI
├── rss_monitor.py              # RSS 크롤링 및 ChromaDB 적재
├── embedding_search_v1.py      # Hybrid search 구현
├── requirements.txt
├── state/
│   └── seen_items.json         # 처리된 RSS 항목 상태
├── chroma_db/                  # 임베딩 벡터 저장소
└── qa_dataset_generation/
    ├── run_qa_pipeline.py      # QA 데이터셋 생성 파이프라인
    ├── train.py                # 파인튜닝 스크립트
    ├── train_guide.ipynb       # 학습 가이드 노트북
    └── data/
        └── qa_dataset.jsonl    # 생성된 QA 데이터셋
```

## 로컬 실행

### 1. Python 3.11 가상환경 생성

```bash
python3.11 -m venv .venv311
```

> Python 3.12에서는 `torch==2.1.0` 호환 문제로 **Python 3.11 권장**

### 2. 의존성 설치

```bash
.venv311/bin/pip install -r requirements.txt
```

### 3. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 필요한 API 키를 설정합니다.

### 4. RSS 모니터 실행

```bash
.venv311/bin/python -u rss_monitor.py
```

- 첫 실행: RSS 최근 50건을 기준 상태로만 저장하고 종료
- 이후 실행: 새 공지가 있을 때만 본문 크롤링 및 임베딩 저장

### 5. Streamlit UI 실행

```bash
.venv311/bin/streamlit run app.py
```

## 크롤링

### 연도별 사용 코드

| 연도 | 파일 | 환경 | 비고 |
|------|------|------|------|
| 2026 | `colab_crawl.py` | Google Colab | OCR·PDF 포함, Drive에서 실행 |
| 2025 (본문) | `crawl_2025.py` | 로컬 | 텍스트 본문만 수집 |
| 2025 (제목) | `crawl_2025_titles.py` | 로컬 | 제목·URL·날짜·카테고리만 수집 |

### 2025년 크롤링 — 2단계

**1단계**: 제목·URL·날짜·카테고리만 수집

```bash
python crawl_2025_titles.py
# 출력: crawl_2025_titles.json
```

**2단계**: 본문 텍스트까지 수집 (OCR·PDF 미포함)

```bash
python crawl_2025.py
# 출력: qa_dataset_generation/data/2025_notice.json
```

### 2026년 크롤링 — Google Colab

`colab_crawl.py`를 Colab에 업로드 후 실행. Drive에 저장된 `2026_notice.json`을 읽어 본문을 재크롤링하며, Clova OCR 및 PDF 추출을 포함합니다.

```
SOURCE_PATH = '/content/drive/MyDrive/sangsangfinder/2026_notice.json'
SAVE_PATH   = '/content/drive/MyDrive/sangsangfinder/2026_notice.json'
```

### 크롤러 모듈 구조

`crawler.py`는 OCR·PDF 추출 함수를 제공하는 **공통 모듈**로, `app.py` 등에서 import해서 사용합니다. `colab_crawl.py`는 이 로직을 포함하여 단독 실행 가능한 Colab용 스크립트로 확장한 버전입니다.

| 기능 | `crawler.py` | `crawl_2025.py` | `colab_crawl.py` |
|------|:---:|:---:|:---:|
| 텍스트 본문 추출 | O | O | O |
| 이미지 OCR (Clova) | O | X | O |
| PDF 추출 | O | X | O |
| 카테고리 분류 | X | O | O |
| 단독 실행 (`__main__`) | X | O | O |

---

## QA 데이터셋 생성

Teacher-Judge 구조로 공지사항 기반 QA 쌍을 자동 생성합니다.

```bash
.venv311/bin/python qa_dataset_generation/run_qa_pipeline.py
```

| 역할 | 모델 | 설명 |
|------|------|------|
| Teacher (QA 생성) | claude-sonnet-4-6 | well-formed + 유저형 QA 쌍 생성 |
| Judge (품질 검증) | gemini-2.5-flash | claim 추출 → hallucination 탐지 → 품질 점수(1~5) |

**Judge 3단계 검증:**
1. **Claim 추출** — 답변에서 날짜·금액·조건 등 사실적 주장 추출
2. **Hallucination 탐지** — 각 주장을 원문 공지와 대조
3. **품질 점수** — 근거 명확성·답변 완성도 평가 (임계값: 4점 이상 통과)

- 입력: `qa_dataset_generation/data/2026_notice.json`
- 출력: `qa_dataset_generation/data/qa_dataset.jsonl` (append 모드, resume 지원)

## 파인튜닝

```bash
.venv311/bin/python qa_dataset_generation/train.py
```

자세한 내용은 [train_guide.ipynb](qa_dataset_generation/train_guide.ipynb) 참고

## GitHub Actions

- 워크플로우: [.github/workflows/rss-monitor.yml](.github/workflows/rss-monitor.yml)
- 10분마다 RSS를 확인하고, 상태 파일과 DB 변경분을 자동 커밋

## 상태 파일

- `state/seen_items.json` — 이미 처리한 RSS 항목 목록
- `chroma_db/` — 임베딩된 공지 벡터 데이터
- `.cache/` — 임베딩 모델 캐시 (첫 실행 시 자동 다운로드)

## 주의사항

- `.env` 파일은 절대 커밋하지 않습니다 (API 키 포함)
- 평가 시 train/test split을 반드시 명시합니다
- 임베딩 모델 최초 다운로드 시 시간이 소요될 수 있습니다
