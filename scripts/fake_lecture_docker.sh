#!/usr/bin/env bash
# 가짜 강의로 검증 탐지율을 테스트하기 위한 헬퍼.
# docker-compose backend/db를 그대로 쓰면서, 앱을 거치지 않고도 lecture_id 기준
# 파일명 규칙(save_upload()가 하는 일)을 처음부터 지켜서 이후 수동 리네임이
# 필요 없게 한다.
#
# 사용법:
#   scripts/fake_lecture_docker.sh preprocess <video_path> [lecture_id]
#     -> 전처리만 실행 (--skip-analyzer). lecture_id를 안 주면 새로 생성.
#        완료 후 merged_clean.json 경로와 lecture_id를 출력한다.
#        이 merged_clean.json을 직접 편집해서 오류를 주입한 다음 아래 verify-only로 넘어간다.
#
#   scripts/fake_lecture_docker.sh verify-only <lecture_id> [title]
#     -> 해당 lecture_id로 Lecture(없으면 생성) + pending ProcessingJob(verify_only)을
#        DB에 넣는다. 워커가 집어가서 전처리는 스킵하고 검증만 새로 돈다.
#        완료되면 GET /lectures/<lecture_id>/result 로 확인 가능.
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE="docker compose"

cmd="${1:-}"

case "$cmd" in
  preprocess)
    video_path="${2:?사용법: $0 preprocess <video_path> [lecture_id]}"
    lecture_id="${3:-$(python3 -c 'import uuid; print(uuid.uuid4())')}"
    if [ ! -f "$video_path" ]; then
      echo "영상 파일을 찾을 수 없습니다: $video_path" >&2
      exit 1
    fi
    ext="${video_path##*.}"
    input_dir="storage/inputs/${lecture_id}"
    mkdir -p "$input_dir"
    dest="${input_dir}/${lecture_id}.${ext}"
    cp "$video_path" "$dest"
    echo "▶ lecture_id: ${lecture_id}"
    echo "▶ 영상 복사: ${dest}  (실제 앱 업로드와 동일하게 lecture_id로 이름을 맞췄습니다)"
    echo "▶ 전처리만 실행 중 (--skip-analyzer, API 비용 없이 검증 단계는 스킵)..."
    $COMPOSE exec -T backend python -m pipeline.main \
      --input "/app/${dest}" \
      --output "/app/storage/results/${lecture_id}" \
      --slides "/app/storage/results/${lecture_id}/slides" \
      --lecture-id "${lecture_id}" \
      --skip-analyzer
    analyzer_dir="storage/results/${lecture_id}/${lecture_id}_analyzer"
    merged_clean="${analyzer_dir}/${lecture_id}_merged_clean.json"
    echo ""
    echo "✓ 전처리 완료"
    echo "  lecture_id   : ${lecture_id}"
    echo "  merged_clean : ${merged_clean}"
    echo ""
    echo "다음 단계: 위 merged_clean.json을 편집해서 오류를 주입한 뒤,"
    echo "  scripts/fake_lecture_docker.sh verify-only ${lecture_id}"
    echo "를 실행하세요."
    ;;

  verify-only)
    lecture_id="${2:?사용법: $0 verify-only <lecture_id> [title]}"
    title="${3:-가짜강의 테스트}"
    ext_glob="storage/inputs/${lecture_id}/${lecture_id}.*"
    video_file=$(ls $ext_glob 2>/dev/null | head -1 || true)
    if [ -z "$video_file" ]; then
      echo "storage/inputs/${lecture_id}/ 에서 ${lecture_id}.<확장자> 영상을 찾을 수 없습니다." >&2
      echo "먼저 '$0 preprocess <video_path>'를 실행했는지 확인하세요." >&2
      exit 1
    fi
    video_rel="inputs/${lecture_id}/$(basename "$video_file")"
    $COMPOSE exec -T backend python -c "
import asyncio, uuid
from app.db import AsyncSessionLocal
from app.models import Lecture, ProcessingJob

async def main():
    lecture_id = uuid.UUID('${lecture_id}')
    async with AsyncSessionLocal() as db:
        existing = await db.get(Lecture, lecture_id)
        if not existing:
            db.add(Lecture(
                id=lecture_id,
                title='${title}',
                video_path='${video_rel}',
                output_dir='results/${lecture_id}',
            ))
        db.add(ProcessingJob(id=uuid.uuid4(), lecture_id=lecture_id, job_type='verify_only', status='pending'))
        await db.commit()
        print('pending verify_only job 등록 완료')

asyncio.run(main())
"
    echo ""
    echo "워커가 몇 초 안에 집어갑니다. 상태 확인:"
    echo "  curl http://localhost:8003/lectures/${lecture_id}"
    echo "결과 확인:"
    echo "  curl http://localhost:8003/lectures/${lecture_id}/result"
    ;;

  *)
    echo "사용법:"
    echo "  $0 preprocess <video_path> [lecture_id]"
    echo "  $0 verify-only <lecture_id> [title]"
    exit 1
    ;;
esac
