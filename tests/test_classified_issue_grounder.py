import pytest

pytest.importorskip("google.genai")

from pipeline.verifier.classified_issue_grounder import _aggregate_grounding_trials


def test_grounding_keeps_only_status_for_insufficient_evidence():
    payload = _aggregate_grounding_trials(
        {"id": "I0001", "category": "factual_error"},
        [
            {
                "status": "insufficient_evidence",
                "claim_verdict": "uncertain",
                "issue_supported": None,
                "reason": "직접 근거가 부족합니다.",
                "evidence_sources": ["https://example.com/weak"],
                "evidence_summary": "긴 요약",
                "search_queries": ["query"],
                "verified_sources": [{"url": "https://example.com/weak"}],
            }
        ],
    )

    assert payload == {"status": "insufficient_evidence"}


def test_grounding_keeps_public_evidence_for_supported_issue():
    payload = _aggregate_grounding_trials(
        {"id": "I0001", "category": "factual_error"},
        [
            {
                "status": "supports_issue",
                "claim_verdict": "claim_false",
                "issue_supported": True,
                "reason": "공식 문서가 claim을 반박합니다.",
                "evidence_summary": "공식 문서에 반대 내용이 있습니다.",
                "source_verification_status": "verified",
                "verified_sources": [
                    {
                        "url": "https://docs.example.com/a",
                        "direct_match": True,
                        "auto_decision_eligible": True,
                        "source_priority": 1,
                        "matched_passages": [
                            {
                                "key_sentence": "The source directly contradicts the lecture claim.",
                                "stance": "supports_issue",
                                "why_relevant": "강의 claim과 직접 충돌합니다.",
                                "match_status": "exact",
                                "match_score": 1.0,
                            }
                        ],
                    }
                ],
                "search_queries": ["query"],
            }
        ],
    )

    assert payload["status"] == "supports_issue"
    assert payload["evidence_sources"] == ["https://docs.example.com/a"]
    assert payload["selected_source_count"] == 1
    assert payload["evidence_passages"][0]["key_sentence"] == "The source directly contradicts the lecture claim."
    assert "claim_verdict" not in payload
    assert "issue_supported" not in payload
    assert "trials" not in payload
    assert "search_queries" not in payload
