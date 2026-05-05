# 파일별 차이
crawler.py — 재사용 가능한 함수 모음. OCR(Clova API), PDF 추출, 이미지 전처리 등 무거운 기능이 여기 있고 다른 스크립트에서 import해서 쓰는 게 원래 의도. 단독 실행 불가.

crawl_2025_titles.py — 목록 페이지만 빠르게 순회해서 제목/URL/날짜만 저장. 본문을 가져오지 않아서 빠름. OCR·PDF 없음.

crawl_2025.py — 목록 순회 + 각 공지 본문까지 한 번에 수집. crawl_2025_titles.py의 완전판이지만 OCR·PDF는 여기서도 없음 (순수 텍스트 추출만). 진행 상황 로그가 더 상세하고 resume 지원.

colab_crawl.py — 이미 만들어진 JSON 파일을 불러와서 각 공지의 본문을 다시 채우는 "보강" 스크립트. crawler.py의 OCR·PDF 기능을 Colab 환경에 맞게 자체 포함시킨 독립 실행형. 카테고리 재분류 기능도 포함.

## Supabase 자동 크롤링

GitHub Actions가 평일 09:17-17:17(KST)에 1시간 간격으로 실행되고, 새 공지만 Supabase Postgres의 `notices` 테이블에 저장한다.

필요한 GitHub Secret:

```text
SUPABASE_DB_URL
```

Supabase에서 DB에 들어간 다음, 상단 DB 이름 옆에 있는 'connect' 연두색 버튼 클릭. Direct > Transaction pooler > Connection string 복사하고 비밀번호 DB 생성 시 정한 거로 변경
GitHub Actions에서는 보통 pooler URI 뒤에 `sslmode=require`가 붙은 값을 사용

로컬 테스트:

```bash
SUPABASE_DB_URL="postgresql://..." python crawling/auto_crawler.py --storage supabase --init-db --no-index --max-pages 5
```

테이블만 수동으로 만들고 싶으면 `crawling/supabase_schema.sql`을 Supabase SQL Editor에서 실행해도 된다.

기존 JSON 파일을 Supabase로 가져오기:

```bash
pip install -r requirements-crawl.txt
SUPABASE_DB_URL="postgresql://..." python scripts/import_json_to_supabase.py data/data_2025.json data/data_2026.json
```

같은 `url`은 `upsert`로 처리되므로 같은 파일을 다시 실행해도 중복 저장되지 않는다.
