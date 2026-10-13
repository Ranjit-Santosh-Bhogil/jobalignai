from __future__ import annotations

import pytest

from rag_pipeline import RequirementAssessment, compute_score, make_documents_from_resume, model_error_message


def test_make_documents_from_resume_creates_chunks() -> None:
    text = "Senior software engineer with Python experience. " * 20
    documents = make_documents_from_resume(text, "resume.pdf")

    assert documents
    for document in documents:
        assert document.page_content.strip()
        assert document.metadata["source"] == "resume.pdf"


def test_compute_score_excludes_needs_review_from_total() -> None:
    assessments = [
        RequirementAssessment(
            requirement="Python",
            category="Skills",
            weight=3,
            status="Met",
            explanation="Strong Python evidence.",
            supporting_evidence=["Python expert"],
            page_references=["Page 1"],
            improvement_suggestion="Keep demonstrating Python work.",
        ),
        RequirementAssessment(
            requirement="Leadership",
            category="Experience",
            weight=2,
            status="Partially Met",
            explanation="Some leadership exposure.",
            supporting_evidence=["Led a project"],
            page_references=["Page 2"],
            improvement_suggestion="Add project leadership examples.",
        ),
        RequirementAssessment(
            requirement="Security clearance",
            category="Compliance",
            weight=5,
            status="Needs Review",
            explanation="Ambiguous evidence.",
            supporting_evidence=[],
            page_references=[],
            improvement_suggestion="Clarify clearance status.",
        ),
    ]

    score = compute_score(assessments)

    assert score["overall_score"] == 80.0
    assert score["needs_review_count"] == 1
    assert score["total_requirements"] == 3


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (ValueError("API key is invalid"), "The provider rejected the authentication request."),
        (ConnectionError("qdrant connection refused"), "Qdrant could not be reached or initialized."),
        (RuntimeError("embedding model failed"), "The embedding request failed."),
    ],
)
def test_model_error_message_maps_known_failures(exception: Exception, expected: str) -> None:
    message = model_error_message(exception)
    assert expected in message
