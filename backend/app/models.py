import uuid
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
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


LECTURE_SOURCE_TAGS = ('youtube', 'kmooc', 'kocw', 'instructor', 'etc')


class Lecture(Base):
    __tablename__ = 'lectures'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    # 업로드 시 필수: youtube | kmooc | kocw | instructor | etc
    source_tag = Column(String, nullable=True)
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
    # Provider-neutral endpoint and stage binding configuration.  Kept
    # separate from the legacy stage_models env map during migration.
    llm_config = Column(JSONB, nullable=False, default=dict, server_default='{}')
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ModelSettingProfile(Base):
    """사용자가 저장한 모델 설정 프리셋."""

    __tablename__ = 'model_setting_profiles'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, unique=True, index=True)
    stage_models = Column(JSONB, nullable=False, default=dict, server_default='{}')
    llm_config = Column(JSONB, nullable=False, default=dict, server_default='{}')
    editor_state = Column(JSONB, nullable=False, default=dict, server_default='{}')
    is_active = Column(Boolean, nullable=False, default=False, server_default='false')
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class LlmCredential(Base):
    """웹에서 등록한 Provider credential.

    API key 원문은 저장하지 않고 application-level encryption으로 암호화한
    값만 저장한다. ``credential_ref``만 모델 설정에 들어가며, 실제 값은
    강의 실행 프로세스에서만 복호화한다.
    """

    __tablename__ = 'llm_credentials'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider = Column(String, nullable=False, index=True)
    model = Column(String, nullable=True)
    fingerprint = Column(String, nullable=False, index=True)
    encrypted_api_key = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class VerificationStats(Base):
    """완료된 verify 실행 1건에서 추출한 통계용 요약. 통계 페이지가 이 테이블만
    집계한다. 원본(진실)은 여전히 디스크의 verification_final.json 등이고, 이 행은
    조회 편의를 위한 투영이다.

    - job_type='verify' + status='done' 인 실행만 적재한다 (verify_only 제외).
    - 재검증 시 같은 lecture_id 행을 지우고 다시 넣어 1강의 1행을 유지한다
      (수정 전후 뷰는 Phase 4에서 정책 재검토).
    - 결과 JSON 파일이 없으면 적재하지 않는다.
    """

    __tablename__ = 'verification_stats'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lecture_id = Column(
        UUID(as_uuid=True), ForeignKey('lectures.id', ondelete='CASCADE'), nullable=False, index=True
    )
    job_id = Column(
        UUID(as_uuid=True), ForeignKey('processing_jobs.id', ondelete='CASCADE'), nullable=True
    )

    # lectures.source_tag 를 비정규화 복사 (집계 시 조인 회피). 없으면 'etc'.
    source_tag = Column(String, nullable=False, server_default='etc')
    # 파이프라인이 분류한 학문 도메인. 불명이면 'etc'(기타).
    domain = Column(String, nullable=False, server_default='etc')
    sub_domain = Column(String, nullable=False, server_default='')

    video_duration_sec = Column(Float, nullable=True)
    preprocess_sec = Column(Float, nullable=True)
    verify_sec = Column(Float, nullable=True)
    total_sec = Column(Float, nullable=True)

    # feedback 상태별 개수. 슬라이드 오류도 지식 오류로 포함한다.
    confirmed_count = Column(Integer, nullable=False, server_default='0')
    review_count = Column(Integer, nullable=False, server_default='0')
    rejected_count = Column(Integer, nullable=False, server_default='0')
    slide_error_count = Column(Integer, nullable=False, server_default='0')

    # 상태 × 유형 분포: { "confirmed": {type: n}, "review": {...}, "rejected": {...} }
    # 통계 페이지는 confirmed+review 만 합산하고 rejected 는 버린다.
    breakdown_by_type = Column(JSONB, nullable=False, default=dict, server_default='{}')

    verifier_models = Column(JSONB, nullable=False, default=list, server_default='[]')
    verifier_version = Column(Integer, nullable=True)

    verification_date = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
