from unittest.mock import patch

from pipeline.verifier.issue_type_classifier import _batch_worker


def _row(issue_id: str, status: str) -> dict:
    return {
        "id": issue_id,
        "status": status,
        "probabilities": {"factual_error": 1.0},
    }


def test_batch_worker_retries_only_unparsed_items():
    calls = []

    def fake_call_model_for_batch(*, model, batch, current_date, max_tokens):
        calls.append([row["id"] for row in batch])
        if len(calls) == 1:
            return [
                _row("I0001", "ok"),
                _row("I0002", "parse_failed"),
                _row("I0003", "ok"),
            ], {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40}
        return [
            _row("I0002", "ok"),
        ], {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}

    batch = [{"id": "I0001"}, {"id": "I0002"}, {"id": "I0003"}]
    resolved = {
        "provider": "anthropic",
        "resolved_model": "claude-test",
    }
    with (
        patch(
            "pipeline.verifier.issue_type_classifier._call_model_for_batch",
            side_effect=fake_call_model_for_batch,
        ),
        patch(
            "pipeline.verifier.issue_type_classifier._resolve_model_spec",
            return_value=resolved,
        ),
        patch.dict(
            "os.environ",
            {
                "ISSUE_TYPE_CLASSIFIER_BATCH_RETRIES": "2",
                "ISSUE_TYPE_CLASSIFIER_BATCH_RETRY_WAIT_SEC": "0",
            },
        ),
    ):
        result = _batch_worker(("claude", batch, 2, 2, "2026-07-22", 1024))

    assert calls == [["I0001", "I0002", "I0003"], ["I0002"]]
    assert [row["id"] for row in result["classifications"]] == [
        "I0001",
        "I0002",
        "I0003",
    ]
    assert all(row["status"] == "ok" for row in result["classifications"])
    assert result["token_usage"] == {
        "input_tokens": 35,
        "output_tokens": 13,
        "reasoning_tokens": 0,
        "tool_input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "total_tokens": 48,
    }


def test_batch_worker_keeps_single_parse_failure_after_retries():
    calls = []

    def fake_call_model_for_batch(*, model, batch, current_date, max_tokens):
        calls.append([row["id"] for row in batch])
        rows = []
        for item in batch:
            status = "parse_failed" if item["id"] == "I0002" else "ok"
            row = _row(item["id"], status)
            row["parse_error"] = "probabilities 합계가 0입니다." if status != "ok" else ""
            rows.append(row)
        return rows, {"total_tokens": 1}

    batch = [{"id": "I0001"}, {"id": "I0002"}]
    resolved = {
        "provider": "anthropic",
        "resolved_model": "claude-test",
    }
    with (
        patch(
            "pipeline.verifier.issue_type_classifier._call_model_for_batch",
            side_effect=fake_call_model_for_batch,
        ),
        patch(
            "pipeline.verifier.issue_type_classifier._resolve_model_spec",
            return_value=resolved,
        ),
        patch.dict(
            "os.environ",
            {
                "ISSUE_TYPE_CLASSIFIER_BATCH_RETRIES": "2",
                "ISSUE_TYPE_CLASSIFIER_BATCH_RETRY_WAIT_SEC": "0",
            },
        ),
    ):
        result = _batch_worker(("claude", batch, 1, 1, "2026-07-22", 1024))

    assert calls == [["I0001", "I0002"], ["I0002"], ["I0002"]]
    assert [row["status"] for row in result["classifications"]] == [
        "ok",
        "parse_failed",
    ]
    assert result["classifications"][1]["parse_error"] == "probabilities 합계가 0입니다."
