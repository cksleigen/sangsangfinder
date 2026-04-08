# Sangsangfinder

한성대학교 공지사항 RSS를 주기적으로 확인하고, 새 글이 올라오면 본문을 크롤링해서 Chroma DB에 적재하는 프로젝트입니다.

## 로컬 실행

### 1. Python 3.11 가상환경 생성

```bash
python3.11 -m venv .venv311
```

### 2. 의존성 설치

```bash
.venv311/bin/pip install -r requirements.txt
```

### 3. RSS 모니터 실행

```bash
.venv311/bin/python -u rss_monitor.py
```

첫 실행에서는 RSS 최근 50건을 기준 상태로만 저장하고 종료합니다.  
이후 실행부터는 새 공지가 있을 때만 본문 크롤링과 임베딩 저장이 진행됩니다.

## 상태 파일

- [state/seen_items.json](/Users/mac/Projects/sangsangfinder/state/seen_items.json)
  - 이미 처리한 RSS 항목 목록을 저장합니다.
- [chroma_db](/Users/mac/Projects/sangsangfinder/chroma_db)
  - 임베딩된 공지 데이터가 저장됩니다.

## 주의사항

- Python 3.12에서는 현재 `torch==2.1.0` 설치가 맞지 않아 로컬 실행 기준으로는 Python 3.11을 권장합니다.
- 임베딩 모델 캐시는 프로젝트 내부 `.cache/` 아래에 저장됩니다.
- 처음 실제 임베딩이 수행될 때는 모델 다운로드 때문에 시간이 조금 걸릴 수 있습니다.

## GitHub Actions

- 워크플로우 파일: [.github/workflows/rss-monitor.yml](/Users/mac/Projects/sangsangfinder/.github/workflows/rss-monitor.yml)
- 10분마다 RSS를 확인하고, 상태 파일과 DB 변경분을 커밋합니다.
