from pipeline.verifier.classified_issue_verifier import (
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
