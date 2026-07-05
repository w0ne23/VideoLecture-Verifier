import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services import lecture_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/results")


@router.get("")
async def list_results(
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=100),
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    scope: str = Query('browse'),
    verified_only: bool = Query(False),
):
    return await lecture_service.list_all_results(
        db, page=page, limit=limit,
        category=category, search=search, scope=scope,
        verified_only=verified_only,
    )


@router.get("/{lecture_id}/timeline")
async def get_timeline(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """강의 타임라인 데이터 조회"""
    return await lecture_service.get_timeline(db, lecture_id)


@router.get("/{lecture_id}/graph")
async def get_knowledge_graph(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """강의 지식 그래프 데이터 조회"""
    return await lecture_service.get_knowledge_graph(db, lecture_id)


@router.get("/{lecture_id}/verifier")
async def get_content_verification(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """강의 verifier 결과 조회"""
    return await lecture_service.get_content_verification(db, lecture_id)


@router.post("/{lecture_id}/query")
async def ask_question(lecture_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    """강의 내용 질의응답 (Lecture ID 기준)"""
    body = await request.json()
    question = body.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    return await lecture_service.ask_question(
        db,
        lecture_id,
        question,
        chat_session_id=body.get("chat_session_id") or "default",
        current_scene_number=body.get("current_scene_number"),
        current_slide_number=body.get("current_slide_number"),
    )


@router.post("/{lecture_id}/graph/enter")
async def graph_enter(lecture_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return await lecture_service.graph_enter(db, lecture_id, session_id)


@router.post("/{lecture_id}/graph/heartbeat")
async def graph_heartbeat(lecture_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return await lecture_service.graph_heartbeat(db, lecture_id, session_id)


@router.post("/{lecture_id}/graph/leave")
async def graph_leave(lecture_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return await lecture_service.graph_leave(db, lecture_id, session_id)


@router.get("/{lecture_id}/graph/status")
async def graph_status(lecture_id: str, db: AsyncSession = Depends(get_db)):
    return await lecture_service.graph_status(db, lecture_id)


@router.get("/{lecture_id}")
async def get_result_detail(lecture_id: str, db: AsyncSession = Depends(get_db)):
    """특정 강의 상세 정보 조회"""
    detail = await lecture_service.get_lecture_detail(db, lecture_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Lecture not found")
    return detail
