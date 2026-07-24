from pipeline.verifier.classified_issue_verifier import (
    build_content_verification_view,
    _composite_candidate_categories,
    _merge_composite_verification,
)


def test_composite_candidate_categories_low_margin_top_two():
    issue = {
        "routing_reasons": ["low_margin"],
        "weighted_scores": {
            "factual_error": 0.38,
            "scope_overclaim": 0.35,
            "temporal_error": 0.27,
        },
    }
    assert _composite_candidate_categories(issue) == ["scope_overclaim", "factual_error"]


def test_composite_candidate_categories_model_disagreement_union():
    issue = {
        "routing_reasons": ["model_disagreement"],
        "model_classifications": [
            {"model": "gpt", "top_issue_type": "factual_error"},
            {"model": "claude", "top_issue_type": "scope_overclaim"},
            {"model": "grok", "top_issue_type": "temporal_error"},
        ],
    }
    assert _composite_candidate_categories(issue) == ["temporal_error", "scope_overclaim", "factual_error"]


def test_merge_composite_verification_uses_weighted_expected_score():
    ref = {
        "id": "I0001",
        "issue": {
            "routing_reasons": ["low_margin"],
            "weighted_scores": {"factual_error": 0.4, "scope_overclaim": 0.35, "temporal_error": 0.25},
        },
    }
    merged = _merge_composite_verification(
        ref,
        ["scope_overclaim", "factual_error"],
        {
            "factual_error": {
                "final_severity_score": 0.5,
                "category": "factual_error",
                "category_label": "사실 오류",
                "model_judgments": [],
            },
            "scope_overclaim": {
                "final_severity_score": 0.8,
                "category": "scope_overclaim",
                "category_label": "과도한 일반화",
                "model_judgments": [],
            },
        },
    )
    assert merged["category"] == "composite_issue"
    assert merged["category_label"] == "복합 오류(과도한 일반화, 사실 오류)"
    assert merged["final_severity_score"] == 0.64
    assert merged["final_severity_percent"] == 64.0
    assert merged["scored_as_composite"] is True
    assert merged["primary_issue_type"] == "scope_overclaim"
    assert merged["composite_scoring"]["normalized_probabilities"] == {
        "scope_overclaim": 0.466667,
        "factual_error": 0.533333,
    }
    assert merged["composite_scoring"]["candidate_scores"] == {
        "scope_overclaim": 0.8,
        "factual_error": 0.5,
    }
    assert merged["composite_scoring"]["candidate_contributions"] == {
        "scope_overclaim": 0.373334,
        "factual_error": 0.266666,
    }
    assert set(merged["candidate_verifications"].keys()) == {"factual_error", "scope_overclaim"}


def test_content_view_omits_duplicate_issue_copies():
    result = {
        "generated_at": "2026-07-24T00:00:00+09:00",
        "model_weights": {"gpt": 1.0},
        "summary": {},
        "all_issues": [
            {
                "id": "I0001",
                "issue_id": "I0001",
                "claim_id": "CL0001",
                "claim_text": "원문",
                "resolved_claim": "정리문",
                "category": "factual_error",
                "category_label": "사실 오류",
                "location": {"slide_number": 1},
                "context": {"context_id": "C1"},
                "final_severity_score": 0.9,
                "model_judgments": [],
            }
        ],
    }
    view = build_content_verification_view(result)
    item = view["feedback_items"][0]

    assert "issues" not in view
    assert "final_confirmed_claims" not in view
    assert "needs_review_claims" not in view
    assert "verifier_rejected_claims" not in view
    assert "source_issues" not in item["evidence"]
    assert item["evidence"]["source_issue_ids"] == ["I0001"]
    assert "professor_feedback" not in item
    assert "all_issues" not in view["views"]["classified_issue_verifier"]
    assert "issues_by_type" not in view["views"]["classified_issue_verifier"]
