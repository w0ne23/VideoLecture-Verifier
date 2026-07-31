import uuid
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


JOB_TYPE_VERIFY = 'verify'
JOB_TYPE_VERIFY_ONLY = 'verify_only'
JOB_TYPE_ALIASES = {
    'verify': JOB_TYPE_VERIFY,
    'verified': JOB_TYPE_VERIFY,
    'verified_upload': JOB_TYPE_VERIFY,
    'run_video': JOB_TYPE_VERIFY,
    'verify_only': JOB_TYPE_VERIFY_ONLY,
    'run_verify': JOB_TYPE_VERIFY_ONLY,
}


def normalize_job_type(value: str | None, default: str = JOB_TYPE_VERIFY) -> str:
    token = (value or default).strip().lower().replace('-', '_')
    return JOB_TYPE_ALIASES.get(token, default)


JOB_STATUS_PENDING = 'pending'
JOB_STATUS_RUNNING = 'running'
JOB_STATUS_DONE = 'done'
JOB_STATUS_ERROR = 'error'
JOB_STATUS_WAITING_APPROVAL = 'waiting_approval'
JOB_STATUS_REJECTED = 'rejected'

RUNNING_STATUSES = {JOB_STATUS_PENDING, JOB_STATUS_RUNNING}
ACTIVE_STATUSES = RUNNING_STATUSES | {JOB_STATUS_WAITING_APPROVAL}


class ProcessingJob(Base):
    __tablename__ = 'processing_jobs'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lecture_id = Column(UUID(as_uuid=True), ForeignKey('lectures.id', ondelete='CASCADE'), nullable=False, index=True)
    job_type = Column(String, nullable=False, default=JOB_TYPE_VERIFY, server_default=JOB_TYPE_VERIFY)
    status = Column(String, nullable=False, default=JOB_STATUS_PENDING)
    current_stage = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    pipeline_stages = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    lecture = relationship('Lecture', back_populates='processing_jobs')


class Lecture(Base):
    __tablename__ = 'lectures'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    video_path = Column(Text, nullable=False)
    output_dir = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    processing_jobs = relationship(
        'ProcessingJob',
        back_populates='lecture',
        order_by='ProcessingJob.created_at',
        cascade='all, delete-orphan',
    )

    @property
    def active_job(self):
        return next((job for job in self.processing_jobs if job.status in ACTIVE_STATUSES), None)

    @property
    def last_job(self):
        return self.processing_jobs[-1] if self.processing_jobs else None


class ModelSettings(Base):
    """관리자가 검증 파이프라인에서 쓸 모델을 지정한 설정. 항상 id=1 한 행만 쓰는
    싱글턴이다. stage_models가 비어있으면 자동 배정(fixed 프로필), 값이 있으면
    직접 설정(generic 프로필)으로 취급한다 — 둘을 나타내는 별도 컬럼을 두지 않아서
    "모드는 auto인데 값은 남아있는" 식의 불일치가 애초에 생기지 않는다."""

    __tablename__ = 'model_settings'

    id = Column(Integer, primary_key=True, default=1)
    stage_models = Column(JSONB, nullable=False, default=dict, server_default='{}')
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
