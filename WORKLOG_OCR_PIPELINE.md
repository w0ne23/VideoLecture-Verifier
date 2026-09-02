# OCR / VLM Merge Worklog

## Current Goal
슬라이드 병합 경계에서 OCR을 먼저 쓰고, 애매한 후보만 VLM으로 넘기는 구조로 정리 중.

## What Was Done

### 1. OCR service was split out
- Added a separate OCR service under `ocr_service/`.
- The OCR service exposes:
  - `GET /health`
  - `POST /ocr`
- Docker compose now includes an `ocr` service with GPU-capable runtime settings.

### 2. Nemotron OCR v2 is wired in
- OCR service builds `nvidia/nemotron-ocr-v2` from the Hugging Face repo.
- Build issues that came up were fixed:
  - repo root was not installable
  - `hatchling.build` was missing under `--no-build-isolation`
  - image path had to be passed as a file path string, not a PIL image
- OCR service is now working and verified with a smoke test.

### 3. OCR smoke test scripts were added
- `./ocr_smoke.sh` in the repo parent is the shortest way to test OCR.
- It calls the OCR service using a sample slide image.
- Verified output shows real text extraction from `scene_007_base.jpg`.

### 4. OCR comparison script was added
- Added `scripts/compare_scene_pair.py`.
- Current behavior:
  - compares `previous scene last annot` vs `next scene base`
  - default scene range is `18..31`
  - uses OCR similarity threshold `0.95`
  - prints `merge / probably_merge / uncertain / reject`
- This was used to inspect scene transitions and confirm the `0.95` threshold is reasonable.

### 5. Pipeline now uses OCR hints
- Added `pipeline/preprocess/ocr_hint.py`.
- `slide_textualizer` now includes OCR hints before sending text to the LLM.
- `local_vlm` also includes OCR hints in its prompt.

### 6. OCR prefilter was added to LocalVLM
- `local_vlm` now prefilters `same_slide_build` candidates using OCR similarity.
- If similarity is `>= 0.95`, the candidate can be accepted before VLM review.
- Lower-similarity candidates still go to VLM.
- The threshold is controlled by:
  - `VLVERIFIER_SLIDE_OCR_SIMILARITY_THRESHOLD=0.83`

## Important Files
- `ocr_service/Dockerfile`
- `ocr_service/app/main.py`
- `pipeline/preprocess/ocr_hint.py`
- `pipeline/preprocess/slide_textualizer.py`
- `pipeline/preprocess/local_vlm.py`
- `scripts/ocr_smoke.py`
- `scripts/compare_scene_pair.py`
- `docker-compose.yml`
- `.env.example`

## Current State
- OCR service runs successfully.
- OCR output is valid for real slide images.
- The comparison script is set to use OCR similarity with `0.95` as the working threshold.
- `local_vlm` has OCR prefilter logic, but the integration still needs a final end-to-end validation pass after rebuilding containers.

## Suggested Next Step
1. Rebuild and restart:
   - `docker compose --profile ocr build backend ocr`
   - `docker compose --profile ocr up -d --force-recreate backend ocr`
2. Run the full pipeline again.
3. Check whether OCR prefilter reduces bad `same_slide_build` merges and leaves only uncertain cases for VLM.
