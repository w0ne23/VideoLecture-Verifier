from __future__ import annotations

import asyncio
import json
import time
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException, Query

from app.config import LITELLM_MODEL_CATALOG_URL

router = APIRouter()

_CACHE_TTL_SECONDS = 3600
_PAGE_SIZE = 500
_MAX_PAGES = 20
_catalog_cache: tuple[float, dict] | None = None


def _provider_label(provider: str) -> str:
    return provider.replace('_', ' ').replace('-', ' ').strip().title()


def _request_json(url: str) -> dict:
    parsed = urlparse(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError('invalid LiteLLM catalog URL')
    request = Request(url, headers={'Accept': 'application/json', 'User-Agent': 'VLVerifier/1.0'})
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode('utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('LiteLLM catalog returned a non-object response')
    return payload


def _normalise_model(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None
    model_id = str(item.get('id') or item.get('model') or '').strip()
    provider = str(item.get('provider') or item.get('litellm_provider') or '').strip()
    if not model_id or not provider:
        return None
    result = {
        'id': model_id,
        'provider': provider,
        'mode': item.get('mode'),
    }
    for key in (
        'max_input_tokens',
        'max_output_tokens',
        'supports_reasoning',
        'supports_vision',
        'supports_function_calling',
        'supports_response_schema',
        'supports_web_search',
        'supports_pdf_input',
    ):
        if key in item:
            result[key] = item[key]
    return result


def _load_catalog() -> dict:
    models: dict[tuple[str, str], dict] = {}
    page = 1
    while page <= _MAX_PAGES:
        query = urlencode({'page': page, 'page_size': _PAGE_SIZE})
        payload = _request_json(f'{LITELLM_MODEL_CATALOG_URL}?{query}')
        for item in payload.get('data', []) if isinstance(payload.get('data'), list) else []:
            model = _normalise_model(item)
            if model:
                models[(model['provider'], model['id'])] = model
        if not payload.get('has_more'):
            break
        page += 1

    grouped: dict[str, list[dict]] = {}
    for model in models.values():
        grouped.setdefault(model['provider'], []).append(model)
    providers = [
        {
            'id': provider,
            'name': _provider_label(provider),
            'model_count': len(provider_models),
        }
        for provider, provider_models in sorted(grouped.items())
    ]
    return {
        'source': 'litellm_model_catalog',
        'catalog_url': LITELLM_MODEL_CATALOG_URL,
        'providers': providers,
        'models': sorted(models.values(), key=lambda item: (item['provider'], item['id'])),
        'fetched_at': int(time.time()),
    }


async def _get_catalog(force_refresh: bool) -> dict:
    global _catalog_cache
    now = time.time()
    if not force_refresh and _catalog_cache and now - _catalog_cache[0] < _CACHE_TTL_SECONDS:
        return _catalog_cache[1]
    try:
        catalog = await asyncio.to_thread(_load_catalog)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f'LiteLLM 모델 카탈로그를 가져오지 못했습니다: {exc}') from exc
    _catalog_cache = (time.time(), catalog)
    return catalog


@router.get('/admin/llm-catalog')
async def get_llm_catalog(refresh: bool = Query(False)):
    return await _get_catalog(refresh)
