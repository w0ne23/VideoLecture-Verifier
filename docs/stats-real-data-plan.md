# 통계 페이지 실데이터 연결 계획

통계 페이지(`frontend/src/pages/StatsPage.jsx`)를 `mockStats.js` 가짜 데이터에서
실제 검증 결과 기반으로 전환한다.

## 배경 — 이미 있는 것

| 데이터 | 위치 | 비고 |
|---|---|---|
| 영상 출처 | `lectures.source_tag` | youtube / kmooc / kocw / instructor / etc — DB에 이미 저장 |
| 도메인 | `..._merged_clean.json`, `..._classified_issue_verifier.json` 의 `domain` / `subdomain` | `engineering` 등 7종, 프론트 `DOMAIN_LABELS` 키와 일치. **최상위 결과 JSON엔 없음** |
| 소요 시간 | `storage/results/{id}/pipeline_timings.json` | `elapsed_total_sec`, 스테이지별(`P*`=전처리 / `V*`=검증), `run_history` |
| 영상 길이 | `..._preprocess_result.json` 의 `duration` (초) | |
| 오류 개수/유형 | `..._verification_final.json` 의 `summary` | `confirmed_feedback_count`, `review_needed_feedback_count`, `rejected_feedback_count`, `slide_error_count`, `breakdown_by_type`, `feedback_items[]` |
| 이슈 유형 taxonomy | `pipeline/verifier/issue_type_classifier.py` | `factual_error / temporal_error / scope_overclaim / confusing_explanation / composite_issue` = 프론트 `ISSUE_TYPES` 와 동일 |

## 현재 아키텍처

- 검증 결과는 **DB에 없음** — 디스크 JSON 파일로만 존재, API가 요청 시 읽어 반환(`lecture_service.get_verified_result`).
- 마이그레이션 도구 없음 — `Base.metadata.create_all` + `backend/app/db.py` 에 수동 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. **새 테이블은 `create_all` 이 자동 생성**.
- 통계 페이지는 API 호출 없이 `mockStats.js` 만 사용.

## 확정된 결정 사항

| 항목 | 결정 |
|---|---|
| 슬라이드 오류 | 지식 오류에 포함 → `composite_issue` 유형으로 typeDist 에 집계 |
| 기각(rejected) | 통계에서 제외. typeDist = **확정 + 교수확인**만 (슬라이드 오류도 확정/교수확인만) |
| 강의 길이 | `video_duration_sec` 실제값 그대로 저장, 상한 없음 |
| `verify_only` | 통계 제외 (개발자용). `job_type = 'verify'` + `status = 'done'` 만 집계 |
| domain 불명 | `etc`(기타) 버킷. `DOMAIN_LABELS` 에 `etc: '기타'` 추가 |
| 결과 파일 없음 | 해당 강의 skip |
| `breakdown_by_type` 저장 형태 | 상태별로 분리 저장: `{ confirmed: {...}, review: {...}, rejected: {...} }` — 프론트가 확정+교수확인 합산, 기각 버림 |

## 전처리 시간 / 검증 시간 정의

`pipeline_timings.json` 의 `timings` 키:

- **전처리 (`P*`)**: `P1A` 슬라이드 추출, `P1B` 오디오 품질, `P2A` 슬라이드 텍스트화, `P2B` 전사, `P3` 오디오 context 후처리
- **검증 (`V*`)**: `V1` analyzer 입력, `V2A` claim 추출, `V2B` 판단, `V2C` 분류, `V2D` 웹 근거, `V2E` 최종 검증

주의: `P1 extract_media total ...`, `V2 run_verifier ...` 같은 **롤업 합계 키**와 하위 키가 같이 들어있음 → 합산 시 이중 카운트 금지. 롤업 키만 쓰거나 하위 키만 쓰거나 통일.
(정확한 키 구조는 실제 full `verify` 1회 실행해서 확정 — Phase 1.6)

---

## Phase 1 — 검증 결과를 DB에 적재

- [x] **1.1** `backend/app/models.py` 에 `VerificationStats` 모델 추가
  ```
  id                  UUID PK
  lecture_id          UUID FK(lectures.id, ondelete=CASCADE), index
  job_id              UUID FK(processing_jobs.id, ondelete=CASCADE)
  source_tag          String        # lectures 에서 비정규화 복사
  domain              String        # 없으면 'etc'
  sub_domain          String
  video_duration_sec  Float
  preprocess_sec      Float
  verify_sec          Float
  total_sec           Float
  confirmed_count     Integer
  review_count        Integer
  rejected_count      Integer
  slide_error_count   Integer
  breakdown_by_type   JSONB         # { confirmed:{type:n}, review:{...}, rejected:{...} }
  verifier_models     JSONB
  verifier_version    Integer
  created_at          DateTime(tz)  server_default now()
  ```
- [x] **1.2** 새 테이블이 `create_all` 로 생성되는지 확인 (백엔드 재기동 후 `\dt`)
- [x] **1.3** `backend/app/services/stats_service.py` 신설
  - `extract_stats(output_dir, lecture, job) -> dict | None`
    - `..._verification_final.json` — `summary`, `feedback_items[]`
    - `pipeline_timings.json` — `preprocess_sec`, `verify_sec`, `total_sec`
    - `..._merged_clean.json` — `domain`, `sub_domain` (없으면 `etc` / `''`)
    - `..._preprocess_result.json` — `video_duration_sec`
    - 파일 없으면 `None` 반환 → 호출부에서 skip
    - `breakdown_by_type`: `feedback_items` 를 `status`(confirmed/professor_check/rejected) × `feedback_type` 로 카운트. 슬라이드 오류(`slide_errors[]` 또는 `composite_issue` feedback)도 상태별로 `composite_issue` 에 합산
- [x] **1.4** `backend/app/worker.py` 완료 지점(`if success:` → `JOB_STATUS_DONE`)에 hook
  - `job_type_val == 'verify'` 일 때만 `stats_service.extract_stats(...)` → `verification_stats` 행 insert
  - 같은 `lecture_id` 재검증 시: 기존 행 삭제 후 재삽입 (현재 상태 기준 1강의 1행)
  - 통계 적재 실패가 job 완료를 막지 않도록 try/except 로 감쌈
- [x] **1.5** 백필 스크립트 `backend/scripts/backfill_verification_stats.py`
  - `storage/results/*` 순회, 완료된 `verify` 강의별로 `extract_stats` → insert
  - 결과 파일 없는 것 skip, 멱등(재실행 시 중복 없음)
- [x] **1.6** 실제 full `verify` 1회 실행 → `pipeline_timings.json` 의 `P*` / `V*` 키 구조 확정, 1.3 롤업 로직 마무리

## Phase 2 — 집계 API

- [x] **2.1** `stats_service.aggregate(db) -> dict`
  ```
  by_tag:      [{ key, label, typeDist, total, lectureCount }]
  by_domain:   [{ key, label, typeDist, total }]
  by_duration: [{ key, label, lectureMin, preprocessMin, verifyMin }]
  ```
  - `typeDist` = confirmed + review 합산 (기각 제외), 5개 유형 전부(`composite_issue` 포함)
  - `by_duration` 버킷: 데이터는 실제 초 그대로, 차트용 구간 경계는 여기서 결정 (예: 0–15 / 15–30 / 30–45 / 45–60 / 60+, `lectureMin` = 구간 내 실제 평균)
  - `verify_only` / 파일없음 행은 애초에 테이블에 없음
- [x] **2.2** `backend/app/routers/stats.py` — `GET /stats`, `main.py` 에 `include_router`
- [x] **2.3** 빈 데이터(강의 0개) 응답 형태 정의

## Phase 3 — 프론트 연결

- [x] **3.1** `frontend/src/data/mockStats.js` → `frontend/src/config/statsConfig.js` 로 이동
  - 유지: `ISSUE_TYPES`, `ISSUE_TYPE_LABELS/COLORS`, `DOMAIN_LABELS`, `DURATION_BUCKETS`, `PROCESS_STAGES`, `rankTypes`, `buildInsight`, `buildDomainInsight`, `buildCompareInsight`
  - 제거: `MOCK_BY_TAG`, `MOCK_BY_DOMAIN`, `MOCK_BY_DURATION`, `MOCK_BEFORE_AFTER_PAIRS`
  - `DOMAIN_LABELS` 에 `etc: '기타'` 추가
- [x] **3.2** `frontend/src/api/stats.js` — `getStats()`
- [x] **3.3** `StatsPage.jsx` — `useQuery(['stats'], getStats)` 로 교체, mock import 제거
- [x] **3.4** 로딩 / 에러 / **빈 상태**(강의 없음) UI
- [x] **3.5** 강의 길이 뷰 컴포넌트(`StackedBarChart`)가 60분 초과 구간을 정상 표시하는지 확인

## Phase 4 — 수정 전후 뷰 (보류)

우선순위 낮음. `BeforeAfterCompare` 컴포넌트와 관련 CSS·`buildCompareInsight` 는
제거했다. 나중에 여건이 되면 재구현한다. 재구현 시 고려할 것:

- "같은 강의 첫 검증 vs 재검증" 페어링 방식
  - A: `pipeline_timings.json` `run_history` 로 첫/마지막 실행 자동 페어
  - B: UI 에 "수정본 재검증" 액션 추가해 명시적 태깅 (`verification_stats` 에 `revision_of` 컬럼)
- `verification_stats` 를 1강의 N행으로 확장하고 Phase 1.4 의 "기존 행 삭제" 정책 재검토

## 하지 않는 것

- Alembic 도입 (새 테이블은 `create_all` 로 충분)
- `docker compose down -v` / DB 재생성 (스키마 추가만, 기존 데이터 유지)
- `verify_only` 결과 집계
