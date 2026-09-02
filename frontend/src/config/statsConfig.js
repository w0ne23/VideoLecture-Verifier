// 통계 페이지 표시 설정 + 인사이트 문구 생성 헬퍼
// 데이터 출처: GET /api/stats (백엔드 verification_stats 집계)

// 지식 오류 유형 — 차트 색상은 전 그래프 공통, composite_issue = 슬라이드 오류
export const ISSUE_TYPES = [
  { key: 'factual_error', label: '사실 오류', color: '#dc2626' },
  { key: 'temporal_error', label: '오래된 내용', color: '#d97706' },
  { key: 'scope_overclaim', label: '과도한 일반화', color: '#2563eb' },
  { key: 'confusing_explanation', label: '혼동 가능 설명', color: '#0d9488' },
  { key: 'composite_issue', label: '슬라이드 오류', color: '#64748b' },
]

export const ISSUE_TYPE_LABELS = Object.fromEntries(ISSUE_TYPES.map(t => [t.key, t.label]))
export const ISSUE_TYPE_COLORS = Object.fromEntries(ISSUE_TYPES.map(t => [t.key, t.color]))

// 파이프라인 도메인 키 → 한글, 불명/미분류는 'etc'(기타)로 유입
export const DOMAIN_LABELS = {
  engineering: '공학',
  natural_science: '자연과학',
  humanities: '인문과학',
  social_science: '사회과학',
  education: '교육학',
  medicine: '의약학',
  arts_sports: '예술체육',
  etc: '기타',
}

// 영상 출처 태그
export const SOURCE_TAG_LABELS = {
  instructor: '교수자 제공',
  kocw: 'KOCW',
  kmooc: 'K-MOOC',
  youtube: 'YouTube',
  etc: '기타',
}

// 강의 길이별 뷰 전용 — 오류 유형이 아니라 파이프라인 소요 시간(전처리/검증) 표시
export const PROCESS_STAGES = [
  { key: 'preprocess', label: '전처리 시간', color: '#0d9488' },
  { key: 'verify', label: '검증 시간', color: '#e11d48' },
]

// typeDist(유형별 건수)를 건수 내림차순으로 정렬, 0건 유형은 제외
export function rankTypes(typeDist) {
  return ISSUE_TYPES
    .map(t => ({ ...t, value: typeDist?.[t.key] || 0 }))
    .filter(t => t.value > 0)
    .sort((a, b) => b.value - a.value)
}

// 뷰별(tag / duration / domain) 인사이트 카드 문구 생성 — { title, bullets[] } 반환
export function buildInsight(view, rows) {
  if (!rows?.length) {
    return { title: '집계된 데이터가 없습니다', bullets: ['검증이 완료된 강의가 아직 없습니다.'] }
  }

  if (view === 'tag') {
    const all = {}
    rows.forEach(r => {
      Object.entries(r.typeDist).forEach(([k, v]) => { all[k] = (all[k] || 0) + v })
    })
    const ranked = rankTypes(all)
    // 출처마다 강의 개수가 달라 총 건수 비교는 왜곡되므로 강의당 평균으로 비교
    const topAvg = rows
      .filter(r => r.lectureCount)
      .map(r => ({ ...r, avg: r.total / r.lectureCount }))
      .sort((a, b) => b.avg - a.avg)[0]
    return {
      title: '강의 영상 출처별 분석 결과 비교',
      bullets: [
        ranked[0]
          ? `전체 오류 중 **${ranked[0].label}**이(가) 가장 많고, 이어서 **${ranked[1]?.label || '—'}** 순이다.`
          : '아직 확정·교수확인된 오류가 없습니다.',
        topAvg
          ? `강의당 평균 오류가 가장 많은 출처는 **${topAvg.label}**(으)로, 강의당 평균 **${topAvg.avg.toFixed(1)}개**가 탐지되었다.`
          : '출처마다 유형 구성이 다릅니다.',
      ],
    }
  }

  if (view === 'duration') {
    const totalPreprocess = rows.reduce((sum, r) => sum + (r.preprocessMin || 0), 0)
    const totalVerify = rows.reduce((sum, r) => sum + (r.verifyMin || 0), 0)
    const totalAll = totalPreprocess + totalVerify
    const totalLectureMin = rows.reduce((sum, r) => sum + (r.lectureMin || 0), 0)
    const secPerMin = totalLectureMin > 0 ? (totalAll / totalLectureMin) * 60 : 0
    const preprocessValues = rows.map(r => r.preprocessMin || 0)
    const verifyValues = rows.map(r => r.verifyMin || 0)
    const preprocessGap = Math.max(...preprocessValues) - Math.min(...preprocessValues)
    const verifyGap = Math.max(...verifyValues) - Math.min(...verifyValues)
    return {
      title: '강의 길이별 파이프라인 소요 시간 경향',
      bullets: [
        `강의 길이가 길어질수록 총 소요 시간도 함께 늘어나며, 평균적으로 강의 1분당 **${secPerMin.toFixed(1)}초**가 소요되었습니다.`,
        `검증 시간은 구간별로 **${verifyGap.toFixed(1)}분** 안팎 차이에 그쳤지만, 전처리 시간은 영상 길이에 따라 **${preprocessGap.toFixed(1)}분**까지 벌어졌습니다.`,
      ],
    }
  }

  const biggest = [...rows].sort((a, b) => b.total - a.total)[0]
  const ranked = rankTypes(biggest.typeDist)
  return {
    title: '강의 영상 도메인별 오류 유형 분포',
    bullets: [
      `**${biggest.label}** 도메인의 오류가 가장 많습니다 (**${biggest.total}건**).`,
      `**${biggest.label}** 안에서는 **${ranked[0]?.label || '—'}** > **${ranked[1]?.label || '—'}** 순입니다.`,
    ],
  }
}

// 도메인별 뷰는 캐러셀이라 현재 표시 중인 도메인 행 하나만 받아 그 기준으로 문구 생성
export function buildDomainInsight(row) {
  if (!row) return { title: '도메인별 오류 유형 분포', bullets: ['표시할 도메인이 없습니다.'] }
  const ranked = rankTypes(row.typeDist)
  return {
    title: `${row.label} 도메인 오류 유형 분포`,
    bullets: [
      `**${row.label}** 도메인에서 오류가 총 **${row.total}건** 탐지되었습니다.`,
      ranked[0]
        ? `가장 많은 유형은 **${ranked[0].label}**이고, 이어서 **${ranked[1]?.label || '—'}** 순입니다.`
        : '이 도메인에서는 아직 집계된 오류가 없습니다.',
    ],
  }
}
