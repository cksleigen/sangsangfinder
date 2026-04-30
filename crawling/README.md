# 파일별 차이
crawler.py — 재사용 가능한 함수 모음. OCR(Clova API), PDF 추출, 이미지 전처리 등 무거운 기능이 여기 있고 다른 스크립트에서 import해서 쓰는 게 원래 의도. 단독 실행 불가.

crawl_2025_titles.py — 목록 페이지만 빠르게 순회해서 제목/URL/날짜만 저장. 본문을 가져오지 않아서 빠름. OCR·PDF 없음.

crawl_2025.py — 목록 순회 + 각 공지 본문까지 한 번에 수집. crawl_2025_titles.py의 완전판이지만 OCR·PDF는 여기서도 없음 (순수 텍스트 추출만). 진행 상황 로그가 더 상세하고 resume 지원.

colab_crawl.py — 이미 만들어진 JSON 파일을 불러와서 각 공지의 본문을 다시 채우는 "보강" 스크립트. crawler.py의 OCR·PDF 기능을 Colab 환경에 맞게 자체 포함시킨 독립 실행형. 카테고리 재분류 기능도 포함.