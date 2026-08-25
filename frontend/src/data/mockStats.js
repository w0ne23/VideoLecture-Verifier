/** 이슈 유형 — 차트 색상은 전 그래프 공통 */
export const ISSUE_TYPES = [
  { key: 'factual_error', label: '사실 오류', color: '#dc2626' },
  { key: 'temporal_error', label: '오래된 내용', color: '#d97706' },
  { key: 'scope_overclaim', label: '과도한 일반화', color: '#2563eb' },
  { key: 'confusing_explanation', label: '혼동 가능 설명', color: '#0d9488' },
  { key: 'composite_issue', label: '슬라이드 오류', color: '#64748b' },
]

export const ISSUE_TYPE_LABELS = Object.fromEntries(ISSUE_TYPES.map(t => [t.key, t.label]))
export const ISSUE_TYPE_COLORS = Object.fromEntries(ISSUE_TYPES.map(t => [t.key, t.color]))

export const DOMAIN_LABELS = {
  engineering: '공학',
  natural_science: '자연과학',
  humanities: '인문과학',
  social_science: '사회과학',
  education: '교육학',
  medicine: '의약학',
  arts_sports: '예술체육',
}

export const DURATION_BUCKETS = [
  { key: '0_15', label: '0–15분' },
  { key: '15_30', label: '15–30분' },
  { key: '30_45', label: '30–45분' },
  { key: '45_60', label: '45분–1시간' },
]

/** typeDist 합 = 확정 + 교수확인 (기각·슬라이드 제외) */
function series(key, label, typeDist, avgProcessMin = null, lectureCount = null) {
  const total = Object.values(typeDist).reduce((sum, n) => sum + n, 0)
  return { key, label, typeDist, total, avgProcessMin, lectureCount }
}

// 출처마다 강의 영상 개수가 달라 총 오류 건수만으로는 비교가 왜곡되므로,
// 강의당 평균 오류 수를 함께 계산할 수 있도록 출처별 강의 개수(lectureCount)를 둔다.
export const MOCK_BY_TAG = [
  series('instructor', '교수자 제공', {
    factual_error: 5, temporal_error: 2, scope_overclaim: 3, confusing_explanation: 6, composite_issue: 1,
  }, null, 12),
  series('kocw', 'KOCW', {
    factual_error: 6, temporal_error: 3, scope_overclaim: 5, confusing_explanation: 4, composite_issue: 2,
  }, null, 9),
  series('kmooc', 'K-MOOC', {
    factual_error: 8, temporal_error: 7, scope_overclaim: 4, confusing_explanation: 3, composite_issue: 1,
  }, null, 6),
  series('youtube', 'YouTube', {
    factual_error: 12, temporal_error: 5, scope_overclaim: 6, confusing_explanation: 4, composite_issue: 2,
  }, null, 3),
]

export const MOCK_BY_DURATION = [
  series('0_15', '0–15분', {
    factual_error: 4, temporal_error: 2, scope_overclaim: 2, confusing_explanation: 2, composite_issue: 1,
  }, 24),
  series('15_30', '15–30분', {
    factual_error: 9, temporal_error: 4, scope_overclaim: 5, confusing_explanation: 4, composite_issue: 1,
  }, 36),
  series('30_45', '30–45분', {
    factual_error: 11, temporal_error: 5, scope_overclaim: 6, confusing_explanation: 3, composite_issue: 2,
  }, 48),
  series('45_60', '45분–1시간', {
    factual_error: 10, temporal_error: 6, scope_overclaim: 4, confusing_explanation: 4, composite_issue: 2,
  }, 62),
]

export const MOCK_BY_DOMAIN = [
  series('engineering', DOMAIN_LABELS.engineering, {
    factual_error: 14, temporal_error: 6, scope_overclaim: 7, confusing_explanation: 5, composite_issue: 2,
  }),
  series('natural_science', DOMAIN_LABELS.natural_science, {
    factual_error: 8, temporal_error: 5, scope_overclaim: 3, confusing_explanation: 3, composite_issue: 1,
  }),
  series('humanities', DOMAIN_LABELS.humanities, {
    factual_error: 5, temporal_error: 3, scope_overclaim: 4, confusing_explanation: 6, composite_issue: 1,
  }),
  series('social_science', DOMAIN_LABELS.social_science, {
    factual_error: 4, temporal_error: 2, scope_overclaim: 5, confusing_explanation: 3, composite_issue: 1,
  }),
  series('education', DOMAIN_LABELS.education, {
    factual_error: 3, temporal_error: 1, scope_overclaim: 2, confusing_explanation: 4, composite_issue: 1,
  }),
  series('medicine', DOMAIN_LABELS.medicine, {
    factual_error: 6, temporal_error: 2, scope_overclaim: 3, confusing_explanation: 2, composite_issue: 1,
  }),
  series('arts_sports', DOMAIN_LABELS.arts_sports, {
    factual_error: 2, temporal_error: 1, scope_overclaim: 1, confusing_explanation: 2, composite_issue: 1,
  }),
]

export function rankTypes(typeDist) {
  return ISSUE_TYPES
    .map(t => ({ ...t, value: typeDist[t.key] || 0 }))
    .filter(t => t.value > 0)
    .sort((a, b) => b.value - a.value)
}

export function buildInsight(view, rows) {
  if (view === 'tag') {
    const all = {}
    rows.forEach(r => {
      Object.entries(r.typeDist).forEach(([k, v]) => { all[k] = (all[k] || 0) + v })
    })
    const ranked = rankTypes(all)
    // 출처마다 강의 영상 개수가 달라 총 건수만 비교하면 왜곡되므로 강의당 평균으로 비교한다.
    const topAvg = rows
      .filter(r => r.lectureCount)
      .map(r => ({ ...r, avg: r.total / r.lectureCount }))
      .sort((a, b) => b.avg - a.avg)[0]
    return {
      title: '강의 영상 출처별 분석 결과 비교',
      bullets: [
        `전체 오류 중 **${ranked[0]?.label}**이(가) 가장 많고, 이어서 **${ranked[1]?.label}** 순이다.`,
        topAvg
          ? `강의당 평균 오류가 가장 많은 출처는 **${topAvg.label}**(으)로, 강의당 평균 **${topAvg.avg.toFixed(1)}개**의 오류가 탐지되었다.`
          : '출처마다 유형 구성이 다릅니다.',
      ],
    }
  }

  if (view === 'duration') {
    const longest = [...rows].sort((a, b) => b.total - a.total)[0]
    const shortest = [...rows].sort((a, b) => a.total - b.total)[0]
    return {
      title: '강의 길이별 오류 탐지 소요 시간 경향',
      bullets: [
        `**${longest.label}** 구간에서 오류가 가장 많이 추출되었습니다 (**${longest.total}건**).`,
        `**${shortest.label}**은(는) 상대적으로 오류가 적습니다 (**${shortest.total}건**).`,
      ],
    }
  }

  const biggest = [...rows].sort((a, b) => b.total - a.total)[0]
  const ranked = rankTypes(biggest.typeDist)
  return {
    title: '강의 영상 도메인별 오류 유형 분포',
    bullets: [
      `**${biggest.label}** 도메인의 오류가 가장 많습니다 (**${biggest.total}건**).`,
      `**${biggest.label}** 안에서는 **${ranked[0]?.label}** > **${ranked[1]?.label || '—'}** 순입니다.`,
    ],
  }
}

function sideTotal(side) {
  return Object.values(side.typeDist).reduce((sum, n) => sum + n, 0)
}

/** 보여주기용 before/after 페어. 나중에 실제 강의에 연결 예정. */
export const MOCK_BEFORE_AFTER_PAIRS = [
  {
    id: 'pair-os-intro',
    title: '운영체제 개론 — 프로세스 스케줄링',
    tagLabel: '교수자 제공',
    domainLabel: '공학',
    durationMin: 28,
    before: {
      label: '수정 전',
      processMin: 41,
      typeDist: { factual_error: 5, temporal_error: 2, scope_overclaim: 2, confusing_explanation: 1, composite_issue: 1 },
    },
    after: {
      label: '수정 후',
      processMin: 38,
      typeDist: { factual_error: 1, temporal_error: 1, scope_overclaim: 1, confusing_explanation: 1, composite_issue: 0 },
    },
  },
  {
    id: 'pair-db-norm',
    title: '데이터베이스 — 정규화',
    tagLabel: 'K-MOOC',
    domainLabel: '공학',
    durationMin: 36,
    before: {
      label: '수정 전',
      processMin: 47,
      typeDist: { factual_error: 6, temporal_error: 2, scope_overclaim: 3, confusing_explanation: 2, composite_issue: 1 },
    },
    after: {
      label: '수정 후',
      processMin: 44,
      typeDist: { factual_error: 2, temporal_error: 1, scope_overclaim: 1, confusing_explanation: 1, composite_issue: 0 },
    },
  },
  {
    id: 'pair-ml-bias',
    title: '머신러닝 — 편향과 분산',
    tagLabel: 'YouTube',
    domainLabel: '자연과학',
    durationMin: 24,
    before: {
      label: '수정 전',
      processMin: 35,
      typeDist: { factual_error: 3, temporal_error: 1, scope_overclaim: 2, confusing_explanation: 1, composite_issue: 1 },
    },
    after: {
      label: '수정 후',
      processMin: 33,
      typeDist: { factual_error: 1, temporal_error: 0, scope_overclaim: 1, confusing_explanation: 0, composite_issue: 0 },
    },
  },
].map(pair => ({
  ...pair,
  before: { ...pair.before, total: sideTotal(pair.before) },
  after: { ...pair.after, total: sideTotal(pair.after) },
}))

export function buildCompareInsight(pair) {
  if (!pair) {
    return { title: '수정 전후', bullets: ['비교할 페어를 선택하세요.'] }
  }
  const delta = pair.after.total - pair.before.total
  const beforeTop = rankTypes(pair.before.typeDist)[0]

  return {
    title: '수정 전후 한눈에',
    bullets: [
      `오류가 **${pair.before.total}건** → **${pair.after.total}건**으로 ${delta <= 0 ? `**${Math.abs(delta)}건** 감소` : `**${delta}건** 증가`}했습니다.`,
      beforeTop ? `수정 전 최다 유형은 **${beforeTop.label}**(**${beforeTop.value}건**)입니다.` : '수정 전 유형 분포를 확인하세요.',
    ],
  }
}
