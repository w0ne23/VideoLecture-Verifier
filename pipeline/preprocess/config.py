from ..config import *  # noqa: F401,F403

from .. import config as _parent_config


# lazy 클라이언트 심볼(gemini_client 등)은 pipeline.config의 __getattr__(PEP 562)로
# 제공되므로 star import에 복사되지 않는다. 여기서 부모 모듈에 위임해 전달한다.
def __getattr__(name: str):
    return getattr(_parent_config, name)
