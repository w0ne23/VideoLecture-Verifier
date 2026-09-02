# 하위 호환용 re-export shim, 실제 구현은 pipeline/verifier/claim_extractor.py 참고
from pipeline.verifier.claim_extractor import *  # noqa: F401,F403
from pipeline.verifier.claim_extractor import (  # noqa: F401
    _claim_extract_batch_mode,
    _claim_extract_context_window,
)
