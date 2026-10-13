from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from config import validate_openai_settings
from rag_pipeline import (
    analyze_resume_against_job,
    create_chat_model,
    create_embedding_model,
    extract_pdf_text,
    model_error_message,
    parse_job_description,
)


st.set_page_config(page_title="JobAlign AI", page_icon="📄", layout="wide")


def _status_badge(status: str) -> str:
    palette = {
        "Met": "#2ecc71",
        "Partially Met": "#f39c12",
        "Not Evidenced": "#e74c3c",
        "Needs Review": "#8e44ad",
    }
    color = palette.get(status, "#95a5a6")
    return f"<span style='color:{color}; font-weight:600;'>{status}</span>"


def _render_requirement_table(requirements: list[dict[str, Any]]) -> None:
    if not requirements:
        st.info("No requirement outcomes were generated.")
        return

    rows = []
    for item in requirements:
        rows.append(
            {
                "Requirement": item["requirement"],
                "Category": item["category"],
                "Weight": item["weight"],
                "Status": item["status"],
                "Explanation": item["explanation"],
                "Evidence": "; ".join(item.get("supporting_evidence", [])[:3]) or "No direct evidence",
                "Improvement": item.get("improvement_suggestion", ""),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_score(score: dict[str, Any]) -> None:
    st.subheader("Match score")
    overall = float(score.get("overall_score", 0.0))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall score", f"{overall:.1f}%")
    c2.metric("Met", score.get("met_count", 0))
    c3.metric("Partially met", score.get("partially_met_count", 0))
    c4.metric("Not evidenced", score.get("not_evidenced_count", 0))

    if score.get("needs_review_count", 0):
        st.caption(f"Needs review: {score.get('needs_review_count', 0)}")


def main() -> None:
    st.title("JobAlign AI")
    st.caption("Compare a resume against a job description and identify strengths, gaps, and interview prep questions.")

    with st.sidebar:
        st.header("Inputs")
        st.write("Upload a PDF resume and paste a job description.")
        st.markdown("---")
        st.write("Tips:")
        st.write("• Use a text-based PDF for best extraction results.")
        st.write("• Include the job description text or a pasted job ad.")

    uploaded_resume = st.file_uploader("Upload resume PDF", type=["pdf"])
    job_text = st.text_area(
        "Paste or type the job description",
        height=220,
        placeholder="Paste the full job description here...",
    )

    if st.button("Analyze resume", type="primary"):
        if uploaded_resume is None:
            st.warning("Please upload a resume PDF before analyzing.")
            return
        if not job_text.strip():
            st.warning("Please paste the job description before analyzing.")
            return

        try:
            settings = validate_openai_settings()
            st.caption(
                f"Configuration loaded from `{settings['ENV_FILE']}` "
                f"using provider `{settings['OPENAI_BASE_URL']}`."
            )
            embeddings = create_embedding_model(
                settings["OPENAI_API_KEY"],
                settings["OPENAI_EMBEDDING_MODEL"],
                settings["OPENAI_BASE_URL"],
            )
            llm = create_chat_model(
                settings["OPENAI_API_KEY"],
                settings["OPENAI_CHAT_MODEL"],
                settings["OPENAI_BASE_URL"],
            )

            resume_bytes = uploaded_resume.read()
            resume_text = extract_pdf_text(resume_bytes, uploaded_resume.name)
            parsed_job = parse_job_description(job_text)

            with st.spinner("Extracting resume content, matching requirements, and generating recommendations..."):
                result = analyze_resume_against_job(
                    resume_text=resume_text,
                    job_text=parsed_job,
                    qdrant_url=settings["QDRANT_URL"],
                    embeddings=embeddings,
                    llm=llm,
                )

            score = result["score"]
            requirements = [item.dict() for item in result["requirements"]]
            recommendations = result["recommendations"]
            questions = [item.dict() for item in result["interview_questions"]]

            _render_score(score)

            st.subheader("Requirement-by-requirement assessment")
            _render_requirement_table(requirements)

            st.subheader("Recommendations")
            if recommendations:
                for recommendation in recommendations:
                    st.markdown(f"- {recommendation}")
            else:
                st.info("No recommendations were generated.")

            st.subheader("Interview prep questions")
            if questions:
                for index, question in enumerate(questions, start=1):
                    st.markdown(f"### {index}. {question['question']}")
                    st.markdown(f"**Category:** {question['category']}")
                    st.markdown(f"**Why this matters:** {question['why_asked']}")
                    st.markdown(f"**Sample answer:** {question['sample_answer']}")
                    st.markdown(f"**Follow-up:** {question['follow_up']}")
                    st.markdown("---")
            else:
                st.info("No interview questions were generated for this profile.")

        except Exception as exc:  # pragma: no cover - UI-level handling
            st.error(model_error_message(exc))
            if st.checkbox("Show technical details"):
                st.exception(exc)


if __name__ == "__main__":
    main()
