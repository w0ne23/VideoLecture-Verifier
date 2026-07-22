import unittest
from unittest.mock import patch

from pipeline.verifier.classified_issue_grounder import ground_detected_issues
from pipeline.verifier.issue_type_classifier import _next_stage_item


class IssueDetectionGroundingTest(unittest.TestCase):
    def test_grounding_enriches_without_filtering(self):
        payload = {
            "issues": [
                {"issue_id": "I0001", "resolved_claim": "claim one"},
                {"issue_id": "I0002", "resolved_claim": "claim two"},
            ]
        }

        def fake_grounding(issue, _current_date, _max_tokens):
            issue_id = issue["issue_id"]
            return (
                {
                    "status": "refutes_issue",
                    "claim_verdict": "claim_true",
                    "issue_supported": False,
                    "reason": f"evidence for {issue_id}",
                    "evidence_sources": [f"https://example.test/{issue_id}"],
                    "evidence_summary": "summary",
                    "search_results": [
                        {
                            "title": "source",
                            "url": f"https://example.test/{issue_id}",
                            "snippet": "snippet",
                            "page_text": "must not propagate",
                        }
                    ],
                },
                {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            )

        with patch(
            "pipeline.verifier.classified_issue_grounder._call_grounding",
            side_effect=fake_grounding,
        ):
            result = ground_detected_issues(
                payload,
                current_date="2026-07-20",
                max_workers=2,
            )

        self.assertEqual([row["issue_id"] for row in result["issues"]], ["I0001", "I0002"])
        self.assertEqual(result["issue_detection_grounding"]["candidate_count"], 2)
        self.assertEqual(result["issue_detection_grounding"]["grounded_count"], 2)
        self.assertEqual(result["issue_detection_grounding"]["token_usage"]["total_tokens"], 24)
        for issue in result["issues"]:
            grounding = issue["pre_grounding"]
            self.assertEqual(grounding["status"], "refutes_issue")
            self.assertNotIn("page_text", grounding["search_results"][0])

    def test_classifier_next_input_keeps_pre_grounding(self):
        grounding = {
            "status": "supports_issue",
            "evidence_sources": ["https://example.test/source"],
        }
        item = _next_stage_item(
            {
                "issue_id": "I0001",
                "claim_id": "CL0001",
                "final_issue_type": "factual_error",
                "pre_grounding": grounding,
            }
        )
        self.assertEqual(item["pre_grounding"], grounding)


if __name__ == "__main__":
    unittest.main()
