"""Encrypted storage and runtime resolution for user-provided LLM credentials."""

from __future__ import annotations

import base64
import hashlib
import os
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import LlmCredential


def mask_secret(value: str) -> str:
    value = str(value or "")
    if len(value) <= 8:
        return value
    return f"{value[:8]}******{value[-4:]}"


def credential_ref(credential: LlmCredential) -> str:
    return f"credential:{credential.id}"


def _cipher() -> Fernet:
    # A dedicated key is preferred. The master key fallback keeps existing local
    # Docker setups usable while allowing production deployments to provide a
    # separate encryption key.
    raw = str(
        os.getenv("VLVERIFIER_CREDENTIAL_ENCRYPTION_KEY", "")
        or os.getenv("LITELLM_MASTER_KEY", "")
        or ""
    ).strip()
    if not raw:
        raise RuntimeError(
            "VLVERIFIER_CREDENTIAL_ENCRYPTION_KEY 또는 LITELLM_MASTER_KEY가 필요합니다."
        )
    try:
        return Fernet(raw.encode("utf-8"))
    except (ValueError, TypeError):
        # Human-readable development secrets (for example sk-...) are converted
        # to a stable Fernet key without storing the original value in the DB.
        derived = base64.urlsafe_b64encode(hashlib.sha256(raw.encode("utf-8")).digest())
        return Fernet(derived)


async def save_credential(
    db: AsyncSession,
    *,
    provider: str,
    model: str,
    api_key: str,
) -> LlmCredential:
    provider = str(provider or "").strip().lower()
    model = str(model or "").strip()[:256]
    api_key = str(api_key or "").strip()
    if not provider:
        raise ValueError("Provider가 필요합니다.")
    if not api_key:
        raise ValueError("API 키가 필요합니다.")
    if len(api_key) > 4096:
        raise ValueError("API 키가 너무 깁니다.")

    fingerprint = hashlib.sha256(f"{provider}\0{api_key}".encode("utf-8")).hexdigest()
    existing = (
        await db.execute(
            select(LlmCredential)
            .where(LlmCredential.provider == provider)
            .where(LlmCredential.fingerprint == fingerprint)
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing:
        if model and not existing.model:
            existing.model = model
        await db.commit()
        await db.refresh(existing)
        return existing

    encrypted = _cipher().encrypt(api_key.encode("utf-8")).decode("ascii")
    row = LlmCredential(
        provider=provider,
        model=model,
        fingerprint=fingerprint,
        encrypted_api_key=encrypted,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


def _parse_ref(value: str) -> UUID | None:
    prefix, _, raw_id = str(value or "").partition(":")
    if prefix != "credential" or not raw_id:
        return None
    try:
        return UUID(raw_id)
    except (TypeError, ValueError):
        return None


def decrypt_credential(row: LlmCredential) -> str:
    try:
        return _cipher().decrypt(row.encrypted_api_key.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"credential을 복호화하지 못했습니다: {row.id}") from exc


async def load_runtime_credentials(db: AsyncSession, refs: set[str]) -> dict[str, str]:
    parsed = {ref: _parse_ref(ref) for ref in refs}
    ids = [value for value in parsed.values() if value is not None]
    if not ids:
        return {}
    rows = (
        await db.execute(select(LlmCredential).where(LlmCredential.id.in_(ids)))
    ).scalars().all()
    by_id = {row.id: row for row in rows}
    result: dict[str, str] = {}
    for ref, row_id in parsed.items():
        if row_id is None:
            continue
        row = by_id.get(row_id)
        if row is not None:
            result[ref] = decrypt_credential(row)
    return result
