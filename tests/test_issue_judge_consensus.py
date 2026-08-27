import json

from pipeline.verifier.run_all import (
    _issue_judge_consensus_decision,
    _write_issue_judge_merged_output,
)


MODELS = ["gpt-5.4", "claude-sonnet-5", "grok-4.5"]


def _decision(issue_models, scores):
    return _issue_judge_consensus_decision(
        "CL0095",
        issue_models=issue_models,
        evaluated_models=MODELS,
        scores_by_claim={"CL0095": scores},
        disagreement_threshold=0.40,
        single_keep_confidence=0.85,
    )


def test_two_model_majority_passes_despite_large_score_spread():
    decision = _decision(
        ["gpt-5.4", "grok-4.5"],
        {"gpt-5.4": 0.83, "claude-sonnet-5": 0.20, "grok-4.5": 0.86},
    )

    assert decision["keep"] is True
    assert decision["status"] == "partial_agreement"
    assert decision["disagreement"]["confidence_delta"] == 0.66
    assert decision["disagreement"]["needs_review"] is True


def test_single_model_requires_strong_confidence():
    weak = _decision(
        ["gpt-5.4"],
        {"gpt-5.4": 0.84, "claude-sonnet-5": 0.20, "grok-4.5": 0.30},
    )
    strong = _decision(
        ["gpt-5.4"],
        {"gpt-5.4": 0.85, "claude-sonnet-5": 0.20, "grok-4.5": 0.30},
    )

    assert weak["keep"] is False
    assert weak["status"] == "rejected_single_model_low_confidence"
    assert strong["keep"] is True
    assert strong["status"] == "single_model_strong"


def test_zero_positive_votes_are_rejected():
    decision = _decision(
        [],
        {"gpt-5.4": 0.79, "claude-sonnet-5": 0.59, "grok-4.5": 0.79},
    )

    assert decision["keep"] is False
    assert decision["status"] == "no_issue"


def test_merged_output_keeps_two_model_majority(tmp_path):
    claim_id = "CL0095"
    judge_results = {
        "gpt-5.4": {
            "ok": True,
            "issues": [{"claim_id": claim_id, "issue": "gpt issue", "confidence": 0.83}],
            "claim_scores": [{"claim_id": claim_id, "confidence": 0.83}],
        },
        "claude-sonnet-5": {
            "ok": True,
            "issues": [],
            "claim_scores": [{"claim_id": claim_id, "confidence": 0.20}],
        },
        "grok-4.5": {
            "ok": True,
            "issues": [{"claim_id": claim_id, "issue": "grok issue", "confidence": 0.86}],
            "claim_scores": [{"claim_id": claim_id, "confidence": 0.86}],
        },
    }

    output_path, payload = _write_issue_judge_merged_output(
        output_dir=tmp_path,
        base_stem="lecture",
        merged_path=tmp_path / "merged.json",
        claims_path=str(tmp_path / "claims.jsonl"),
        models=MODELS,
        judge_results=judge_results,
    )

    assert [issue["claim_id"] for issue in payload["issues"]] == [claim_id]
    assert payload["issues"][0]["detected_by_models"] == ["gpt-5.4", "grok-4.5"]
    assert payload["issues"][0]["detector_consensus"]["status"] == "partial_agreement"
    assert payload["summary"]["rejected_single_model_low_confidence_count"] == 0
    assert json.loads((tmp_path / "lecture_issue_judge.json").read_text())["issues"][0]["claim_id"] == claim_id
    assert output_path == str(tmp_path / "lecture_issue_judge.json")
