import sys
from pathlib import Path

# 백엔드는 'app.*', 파이프라인은 'pipeline.*'로 import되므로 둘 다 경로에 추가
ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / 'backend'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
