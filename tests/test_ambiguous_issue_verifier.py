from pipeline.verifier.classified_issue_verifier import (
    ISSUE_TYPES,
    _ambiguous_candidate_categories,
    _merge_ambiguous_verification,
)


def test_ambiguous_candidate_categories_low_margin_top_two():
    issue = {
        "routing_reasons": ["low_margin"],
        "weighted_scores": {
            "factual_error": 0.38,
            "scope_overclaim": 0.35,
            "temporal_error": 0.27,
        },
    }
    assert _ambiguous_candidate_categories(issue) == ["scope_overclaim", "factual_error"]


def test_ambiguous_candidate_categories_model_disagreement_union():
    issue = {
        "routing_reasons": ["model_disagreement"],
        "model_classifications": [
            {"model": "gpt", "top_issue_type": "factual_error"},
            {"model": "claude", "top_issue_type": "scope_overclaim"},
            {"model": "grok", "top_issue_type": "temporal_error"},
        ],
    }
    assert _ambiguous_candidate_categories(issue) == list(ISSUE_TYPES)


def test_merge_ambiguous_verification_picks_highest_score():
    ref = {
        "id": "I0001",
        "issue": {
            "routing_reasons": ["low_margin"],
            "weighted_scores": {"factual_error": 0.4, "scope_overclaim": 0.35, "temporal_error": 0.25},
        },
    }
    merged = _merge_ambiguous_verification(
        ref,
        ["factual_error", "scope_overclaim"],
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
    assert merged["selected_issue_type"] == "scope_overclaim"
    assert merged["selected_from_ambiguous"] is True
    assert set(merged["candidate_verifications"].keys()) == {"factual_error", "scope_overclaim"}
