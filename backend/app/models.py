import uuid
from sqlalchemy import Boolean, Column, String, Text, DateTime, ForeignKey, UniqueConstraint, Integer, Float, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


JOB_TYPE_LEGACY_FULL = "legacy_full"
JOB_TYPE_VERIFY = "verify"
JOB_TYPE_UPLOAD = "upload"
JOB_TYPE_PUBLISH = "publish"
JOB_TYPE_DIRECT_UPLOAD = "direct_upload"
JOB_TYPE_VERIFIED_UPLOAD = "verified_upload"
JOB_TYPE_GRAPH_UPLOAD = "graph_upload"
JOB_TYPE_ALIASES = {
    "legacy": JOB_TYPE_LEGACY_FULL,
    JOB_TYPE_LEGACY_FULL: JOB_TYPE_LEGACY_FULL,
    "verified": JOB_TYPE_VERIFY,
    JOB_TYPE_VERIFIED_UPLOAD: JOB_TYPE_VERIFY,
    JOB_TYPE_VERIFY: JOB_TYPE_VERIFY,
    "publication": JOB_TYPE_PUBLISH,
    JOB_TYPE_PUBLISH: JOB_TYPE_PUBLISH,
    "direct": JOB_TYPE_PUBLISH,
    JOB_TYPE_DIRECT_UPLOAD: JOB_TYPE_PUBLISH,
    "graph": JOB_TYPE_PUBLISH,
    JOB_TYPE_GRAPH_UPLOAD: JOB_TYPE_PUBLISH,
    JOB_TYPE_UPLOAD: JOB_TYPE_PUBLISH,
}


def normalize_job_type(value: str | None, default: str = JOB_TYPE_LEGACY_FULL) -> str:
    token = (value or default).strip().lower().replace("-", "_")
    return JOB_TYPE_ALIASES.get(token, default)

JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_DONE = "done"
JOB_STATUS_ERROR = "error"
JOB_STATUS_WAITING_APPROVAL = "waiting_approval"
JOB_STATUS_REJECTED = "rejected"

WORKER_RUNNABLE_STATUSES = {JOB_STATUS_PENDING}
RUNNING_STATUSES = {JOB_STATUS_PENDING, JOB_STATUS_RUNNING}
ACTION_REQUIRED_STATUSES = {JOB_STATUS_WAITING_APPROVAL}
ACTIVE_STATUSES = RUNNING_STATUSES | ACTION_REQUIRED_STATUSES


class ProcessingJob(Base):
    """파이프라인 실행 로그 — Lecture당 N개 (다대일)"""
    __tablename__ = "processing_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lecture_id = Column(
        UUID(as_uuid=True),
        ForeignKey("lectures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_type = Column(String, nullable=False, default=JOB_TYPE_LEGACY_FULL, server_default=JOB_TYPE_LEGACY_FULL)
    status = Column(String, nullable=False, default=JOB_STATUS_PENDING)
    current_stage = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    pipeline_stages = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    lecture = relationship("Lecture", back_populates="processing_jobs")


class Lecture(Base):
    """강의 영상의 영구 데이터 — 분석과 무관하게 항상 존재"""
    __tablename__ = "lectures"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=True)
    category = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    is_verified = Column(Boolean, nullable=False, default=False, server_default="false")
    is_published = Column(Boolean, nullable=False, default=False, server_default="false")
    video_path = Column(Text, nullable=False)   # inputs/{lecture_id}/{filename}
    output_dir = Column(Text, nullable=False)   # results/{lecture_id}/
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    processing_jobs = relationship(
        "ProcessingJob",
        back_populates="lecture",
        order_by="ProcessingJob.created_at",
    )

    @property
    def active_job(self):
        """현재 실행 중인 job — 최대 1개"""
        return next(
            (j for j in self.processing_jobs if j.status in ACTIVE_STATUSES),
            None,
        )

    @property
    def last_job(self):
        return self.processing_jobs[-1] if self.processing_jobs else None


class LectureMetadata(Base):
    """추천/필터용 강의 메타데이터 요약본.

    전체 원본 metadata JSON은 파일로 유지하고, DB에는 추천 런타임에서 자주 쓰는
    구조화 필드만 저장한다.
    """
    __tablename__ = "lecture_metadata"
    __table_args__ = (
        Index("idx_lm_domain", "domain"),
        Index("idx_lm_difficulty", "difficulty"),
        Index("idx_lm_keywords_gin", "keywords", postgresql_using="gin"),
    )

    lecture_id = Column(
        UUID(as_uuid=True),
        ForeignKey("lectures.id", ondelete="CASCADE"),
        primary_key=True,
    )
    title = Column(Text, nullable=False)
    instructor_id = Column(Text, nullable=True)
    domain = Column(Text, nullable=True)
    graph_domain = Column(Text, nullable=True)
    graph_subdomain = Column(Text, nullable=True)
    difficulty = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    learning_objectives = Column(ARRAY(Text), nullable=True)
    keywords = Column(JSONB, nullable=True)
    concept_roles = Column(JSONB, nullable=True)
    concept_relations = Column(JSONB, nullable=True)
    communities = Column(JSONB, nullable=True)
    visual_concept_terms = Column(ARRAY(Text), nullable=True)
    pedagogy = Column(JSONB, nullable=True)
    diagnostics = Column(JSONB, nullable=True)
    duration_sec = Column(Float, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), nullable=True)
    metadata_version = Column(Integer, nullable=False, default=1)
    metadata_uri = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class GraphSession(Base):
    """강의별 Neo4j 로드 상태 관리 — lecture_id로 식별"""
    __tablename__ = "graph_sessions"
    __table_args__ = (
        UniqueConstraint("lecture_id", "session_id", name="uq_graph_sessions_lecture_session"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lecture_id = Column(
        UUID(as_uuid=True),
        ForeignKey("lectures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stem = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    last_heartbeat_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ChatSession(Base):
    """강의별 QnA 대화 세션."""
    __tablename__ = "chat_sessions"
    __table_args__ = (
        UniqueConstraint("lecture_id", "session_id", name="uq_chat_sessions_lecture_session"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lecture_id = Column(
        UUID(as_uuid=True),
        ForeignKey("lectures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ChatMessage(Base):
    """사용자 질의와 응답 로그. 통계/추천/멀티턴 컨텍스트의 원천 데이터."""
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lecture_id = Column(
        UUID(as_uuid=True),
        ForeignKey("lectures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chat_session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turn_index = Column(Integer, nullable=False, default=0)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    query_major = Column(String, nullable=False, default="A")
    query_minor = Column(String, nullable=False, default="1")
    query_type_label = Column(String, nullable=False, default="내용 질의/개념")
    source_mode = Column(String, nullable=True)
    related_slides = Column(JSONB, nullable=True)
    retrieved_chunks = Column(JSONB, nullable=True)
    core_graph = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
