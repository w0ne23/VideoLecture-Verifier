# DB ORM 모델과 관련 상수 정의
import uuid
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# 파이프라인 실행 종류: 전체 검증(verify) / 검증만 재실행(verify_only)
JOB_TYPE_VERIFY = 'verify'
JOB_TYPE_VERIFY_ONLY = 'verify_only'
# 과거/외부에서 쓰이던 job_type 표기를 정규 값으로 매핑
JOB_TYPE_ALIASES = {
    'verify': JOB_TYPE_VERIFY,
    'verified': JOB_TYPE_VERIFY,
    'verified_upload': JOB_TYPE_VERIFY,
    'run_video': JOB_TYPE_VERIFY,
    'verify_only': JOB_TYPE_VERIFY_ONLY,
    'run_verify': JOB_TYPE_VERIFY_ONLY,
}


# job_type 별칭을 정규 값으로 변환, 별칭이 아니면 default 반환
def normalize_job_type(value: str | None, default: str = JOB_TYPE_VERIFY) -> str:
    token = (value or default).strip().lower().replace('-', '_')
    return JOB_TYPE_ALIASES.get(token, default)


# 처리 작업 상태 값
JOB_STATUS_PENDING = 'pending'
JOB_STATUS_RUNNING = 'running'
JOB_STATUS_DONE = 'done'
JOB_STATUS_ERROR = 'error'
JOB_STATUS_WAITING_APPROVAL = 'waiting_approval'
JOB_STATUS_REJECTED = 'rejected'

# 실행 중으로 취급하는 상태 그룹 / 승인 대기까지 포함한 활성 상태 그룹
RUNNING_STATUSES = {JOB_STATUS_PENDING, JOB_STATUS_RUNNING}
ACTIVE_STATUSES = RUNNING_STATUSES | {JOB_STATUS_WAITING_APPROVAL}


# 강의별 파이프라인 실행 1건을 나타내는 테이블
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


# 업로드된 강의 원본 정보와 연관된 처리 작업 목록을 나타내는 테이블
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

    # 진행 중이거나 승인 대기 중인 작업, 없으면 None
    @property
    def active_job(self):
        return next((job for job in self.processing_jobs if job.status in ACTIVE_STATUSES), None)

    # 가장 최근 생성된 작업, 없으면 None
    @property
    def last_job(self):
        return self.processing_jobs[-1] if self.processing_jobs else None


class ModelSettings(Base):
    """관리자가 검증 파이프라인에서 쓸 모델을 지정하는 설정, id=1 한 행만 쓰는 싱글턴
    stage_models가 비어있으면 자동 배정(fixed 프로필), 값이 있으면 직접 설정(generic 프로필)으로 취급
    모드/값을 나타내는 별도 컬럼을 두지 않아 "모드는 auto인데 값은 남아있는" 식의 불일치를 원천 차단"""

    __tablename__ = 'model_settings'

    id = Column(Integer, primary_key=True, default=1)
    stage_models = Column(JSONB, nullable=False, default=dict, server_default='{}')
    # provider 중립적인 endpoint/stage 바인딩 설정, 마이그레이션 기간 동안 레거시 stage_models와 분리 유지
    llm_config = Column(JSONB, nullable=False, default=dict, server_default='{}')
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ModelSettingProfile(Base):
    """사용자가 저장한 모델 설정 프리셋"""

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
    """웹에서 등록한 provider credential

    API key 원문은 저장하지 않고 application-level encryption으로 암호화한 값만 저장
    credential_ref만 모델 설정에 들어가고 실제 값은 강의 실행 프로세스에서만 복호화
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
    """완료된 verify 실행 1건에서 추출한 통계용 요약, 통계 페이지는 이 테이블만 집계
    원본은 여전히 디스크의 verification_final.json 등이고 이 행은 조회 편의용 투영

    - job_type='verify' + status='done' 실행만 적재 (verify_only 제외)
    - 재검증 시 같은 lecture_id 행 삭제 후 재삽입, 1강의 1행 유지 (수정 전후 뷰는 Phase 4에서 정책 재검토)
    - 결과 JSON 파일 없으면 미적재
    """

    __tablename__ = 'verification_stats'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lecture_id = Column(
        UUID(as_uuid=True), ForeignKey('lectures.id', ondelete='CASCADE'), nullable=False, index=True
    )
    job_id = Column(
        UUID(as_uuid=True), ForeignKey('processing_jobs.id', ondelete='CASCADE'), nullable=True
    )

    # lectures.source_tag 비정규화 복사 (집계 시 조인 회피), 없으면 'etc'
    source_tag = Column(String, nullable=False, server_default='etc')
    # 파이프라인이 분류한 학문 도메인, 불명이면 'etc'(기타)
    domain = Column(String, nullable=False, server_default='etc')
    sub_domain = Column(String, nullable=False, server_default='')

    video_duration_sec = Column(Float, nullable=True)
    preprocess_sec = Column(Float, nullable=True)
    verify_sec = Column(Float, nullable=True)
    total_sec = Column(Float, nullable=True)

    # feedback 상태별 개수, 슬라이드 오류도 지식 오류로 포함
    confirmed_count = Column(Integer, nullable=False, server_default='0')
    review_count = Column(Integer, nullable=False, server_default='0')
    rejected_count = Column(Integer, nullable=False, server_default='0')
    slide_error_count = Column(Integer, nullable=False, server_default='0')

    # 상태 × 유형 분포: { "confirmed": {type: n}, "review": {...}, "rejected": {...} }
    # 통계 페이지는 confirmed+review만 합산, rejected는 제외
    breakdown_by_type = Column(JSONB, nullable=False, default=dict, server_default='{}')

    verifier_models = Column(JSONB, nullable=False, default=list, server_default='[]')
    verifier_version = Column(Integer, nullable=True)

    verification_date = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
