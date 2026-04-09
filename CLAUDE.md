# CLAUDE.md — sangsangfinder

## Project Overview
Hansung University notice search & recommendation system using RAG pipeline.
- Embedding: `jhgan/ko-sroberta-multitask`
- Hybrid search: BM25 (0.3) + dense vector (0.7)
- Vector DB: ChromaDB → Pinecone 마이그레이션 예정
- UI: Streamlit
- Deployment: Streamlit Community Cloud 예정 (변경 가능)

---

## Claude Behavior (Token Efficiency)

- 행동 전 먼저 생각하라. 코드 작성 전 기존 파일을 읽어라.
- 출력은 간결하게, 추론은 철저하게.
- 파일 전체 재작성보다 부분 편집(Edit)을 우선하라.
- 이미 읽은 파일은 변경 가능성이 없으면 다시 읽지 마라.
- 완료 선언 전 코드를 테스트하라.
- 서두 인사말("물론이죠!", "좋은 질문입니다!") 및 마무리 멘트("도움이 됐으면 좋겠습니다") 금지.
- 해결책은 단순하고 직접적으로. 불필요한 추상화 금지.
- 사용자 지시는 이 파일보다 항상 우선한다.

---

## Core Principles

### 1. Logical Consistency Check (논리적 정합성)
모든 코드 변경 전 반드시 확인:
- 데이터 흐름이 앞 단계 → 뒤 단계로 일관되게 이어지는가?
- 전처리 → 임베딩 → 인덱싱 → 검색 → 생성 각 단계의 입출력 타입이 맞는가?
- 새 모듈이 기존 인터페이스를 깨지 않는가?

### 2. Evaluation Data Integrity (평가 데이터 격리)
**절대 규칙: test/val 셋은 학습·파인튜닝·파라미터 튜닝에 절대 사용하지 않는다.**
- 평가 전 반드시 split 출처를 명시적으로 주석으로 표기
```python
  # TRAIN split — used for tuning BM25 k1/b parameters
  # TEST split  — held out, touched only in final eval
```
- 평가 스크립트는 항상 split 출처를 로그에 출력
- 지표(Recall@K, MRR)를 보고할 때 어떤 split인지 반드시 명시

### 3. TDD (Test-Driven Development)
새 기능 구현 순서:
1. `tests/` 에 실패하는 테스트 먼저 작성
2. 테스트를 통과하는 최소 구현
3. 리팩토링
```
tests/
test_chunker.py       # 청킹 로직
test_retriever.py     # hybrid search 결과 검증
test_reranker.py      # reranker context compression
test_eval_pipeline.py # split 격리 검증 포함
```

테스트 실행:
```bash
pytest tests/ -v --tb=short
```

---

## Workflow Rules

- **새 컴포넌트 추가 시**: 테스트 → 구현 → 문서 순서 준수
- **평가 지표 보고 시**: split 명시 없이는 숫자만 말하지 않기
- **파이프라인 변경 시**: 상·하류 단계 인터페이스 영향 먼저 확인

---

## Security

- **`.env` 파일은 절대 열람·출력·커밋하지 않는다.** API 키 등 민감 정보가 포함되어 있으므로, 어떤 이유로도 내용을 읽거나 로그에 출력하지 않는다.

---

## Known Pitfalls
- [ ] 과거 사례: test/val 분리 없이 전체 데이터로 Recall@K 계산 → 지표 신뢰 불가
      → 방지책: `test_eval_pipeline.py`에서 split 격리 assertion 필수