from __future__ import annotations

import json
import re
import uuid
from typing import Any, Literal

import fitz
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


class RequirementSpec(BaseModel):
    name: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    weight: int = Field(..., ge=1, le=5)


class RequirementAssessment(BaseModel):
    requirement: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    weight: int = Field(..., ge=1, le=5)
    status: Literal["Met", "Partially Met", "Not Evidenced", "Needs Review"]
    explanation: str = Field(..., min_length=1)
    supporting_evidence: list[str] = Field(default_factory=list)
    page_references: list[str] = Field(default_factory=list)
    improvement_suggestion: str = Field(..., min_length=1)


class InterviewQuestion(BaseModel):
    question: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    why_asked: str = Field(..., min_length=1)
    sample_answer: str = Field(..., min_length=1)
    follow_up: str = Field(..., min_length=1)


class JobAnalysisResult(BaseModel):
    overall_score: float = Field(default=0.0)
    total_requirements: int = 0
    met_count: int = 0
    partially_met_count: int = 0
    not_evidenced_count: int = 0
    needs_review_count: int = 0
    requirements: list[RequirementAssessment] = Field(default_factory=list)
    interview_questions: list[InterviewQuestion] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sanitize_paragraphs(text: str) -> str:
    chunks = [part.strip() for part in text.split("\n") if part.strip()]
    cleaned: list[str] = []
    for chunk in chunks:
        cleaned.append(re.sub(r"\s+", " ", chunk))
    return "\n\n".join(cleaned)


def extract_pdf_text(file_bytes: bytes, file_name: str) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Could not open '{file_name}' as a PDF. The file may be scanned or not a valid PDF.") from exc

    text_parts: list[str] = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")
        if text and text.strip():
            text_parts.append(f"[Page {page_num + 1}]\n{text.strip()}")

    doc.close()
    combined = "\n\n".join(text_parts)
    if not combined.strip():
        raise ValueError(f"No extractable text was found in '{file_name}'. Please upload a text-based PDF.")
    return combined


def parse_job_description(job_text: str) -> str:
    cleaned = sanitize_paragraphs(job_text)
    if not cleaned or not cleaned.strip():
        raise ValueError("The job description is empty. Please paste or upload valid text.")
    return cleaned


def make_documents_from_resume(resume_text: str, source_name: str) -> list[Document]:
    cleaner = sanitize_paragraphs(resume_text)
    sections = [section.strip() for section in cleaner.split("\n\n") if section.strip()]
    if not sections:
        raise ValueError("No text was extracted from the resume. Please upload a valid PDF.")

    documents: list[Document] = []
    for index, section in enumerate(sections):
        page_marker_match = re.search(r"\[Page (\d+)\]", section)
        page_number = int(page_marker_match.group(1)) if page_marker_match else 1
        doc_text = re.sub(r"\[Page \d+\]", "", section).strip()
        if not doc_text:
            continue
        documents.append(
            Document(
                page_content=doc_text,
                metadata={"source": source_name, "page": page_number, "chunk_id": f"resume-{index}"},
            )
        )

    if not documents:
        raise ValueError("The resume did not produce any usable content after cleaning.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120, separators=["\n\n", "\n", ". ", " "])
    chunks = splitter.split_documents(documents)
    valid_chunks: list[Document] = []
    for i, chunk in enumerate(chunks):
        text = (chunk.page_content or "").strip()
        if not text:
            continue
        chunk.metadata = {
            "source": source_name,
            "page": chunk.metadata.get("page", 1),
            "chunk_id": f"resume-chunk-{i}",
        }
        valid_chunks.append(chunk)
    if not valid_chunks:
        raise ValueError("The resume produced no valid chunk content for analysis.")
    return valid_chunks


def create_embedding_model(api_key: str, model_name: str, base_url: str) -> OpenAIEmbeddings:
    try:
        return OpenAIEmbeddings(
            api_key=api_key,
            model=model_name,
            base_url=base_url,
            # Third-party model names are often unknown to tiktoken. Let the
            # provider perform tokenization instead of failing locally.
            tiktoken_enabled=False,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unable to initialize OpenAI embeddings: {exc}") from exc


def create_chat_model(api_key: str, model_name: str, base_url: str) -> ChatOpenAI:
    try:
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unable to initialize OpenAI chat model: {exc}") from exc


def ping_qdrant(qdrant_url: str) -> QdrantClient:
    client = QdrantClient(url=qdrant_url, timeout=10)
    try:
        client.get_collections()
        return client
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Qdrant is unavailable at {qdrant_url}. Start Docker with 'docker compose up -d'.") from exc


def get_collection_name() -> str:
    return f"resume_match_{uuid.uuid4().hex}"


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    try:
        existing = client.get_collection(collection_name)
        if existing:
            return
    except Exception:
        pass

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Could not create the temporary Qdrant collection '{collection_name}'.") from exc


def index_resume_chunks(chunks: list[Document], embeddings: OpenAIEmbeddings, qdrant_url: str, collection_name: str) -> None:
    client = ping_qdrant(qdrant_url)
    try:
        vector_size = len(embeddings.embed_query("test vector"))
    except Exception as exc:
        raise RuntimeError(
            "The embedding provider request failed. "
            "Check OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL, and provider embedding support."
        ) from exc
    ensure_collection(client, collection_name, vector_size)
    try:
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=qdrant_url,
            collection_name=collection_name,
            prefer_grpc=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Embedding or indexing failed while storing resume chunks in Qdrant. "
            "Check the provider embedding endpoint and model."
        ) from exc


def delete_collection(collection_name: str, qdrant_url: str) -> None:
    if not collection_name:
        return
    try:
        client = QdrantClient(url=qdrant_url, timeout=10)
        client.delete_collection(collection_name=collection_name)
    except Exception:
        pass


def extract_requirements(job_text: str, llm: ChatOpenAI) -> list[RequirementSpec]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You extract 5 to 12 concise job requirements from the job description. "
                "Return only valid JSON with an array of objects. Each object includes: "
                "name, category, weight. The weight is 1 to 5 and reflects importance. "
                "Focus on skills, experience, responsibilities, tools, education, and certifications. "
                "Deduplicate and ensure each requirement is distinct.",
            ),
            ("human", "Job description:\n{job_text}"),
        ]
    )
    structured_llm = llm.with_structured_output(list[RequirementSpec])
    try:
        response = structured_llm.invoke(prompt.format(job_text=job_text))
    except Exception as exc:  # pragma: no cover
        raise ValueError("The model could not parse the job description into structured requirements.") from exc

    if not response:
        raise ValueError("No job requirements were extracted. Please provide a more detailed job description.")
    return response


def retrieve_evidence(resume_store: QdrantVectorStore, requirement: str, top_k: int = 4) -> list[Document]:
    try:
        return resume_store.similarity_search(requirement, k=top_k)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Evidence retrieval failed for requirement '{requirement}'.") from exc


def assess_requirement(requirement: RequirementSpec, evidence_docs: list[Document], llm: ChatOpenAI) -> RequirementAssessment:
    evidence_text = "\n\n".join(f"- {doc.page_content.strip()}" for doc in evidence_docs if doc.page_content and doc.page_content.strip())
    if not evidence_text.strip():
        raise ValueError(f"No evidence was retrieved for requirement '{requirement.name}'.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Assess whether the candidate meets the requirement using only the supplied evidence. "
                "Return valid JSON with fields: requirement, category, weight, status, explanation, "
                "supporting_evidence, page_references, improvement_suggestion. "
                "Status must be one of: Met, Partially Met, Not Evidenced, Needs Review. "
                "Never invent facts. If the evidence is insufficient or ambiguous, use Needs Review. "
                "Do not claim the candidate has experience that is not supported by the evidence. "
                "Use supporting_evidence as short quote excerpts. "
                "Page references should be strings like 'Page 3'.",
            ),
            (
                "human",
                "Requirement: {requirement}\nCategory: {category}\nWeight: {weight}\nEvidence:\n{evidence}",
            ),
        ]
    )
    structured_llm = llm.with_structured_output(RequirementAssessment)
    try:
        response = structured_llm.invoke(
            prompt.format(
                requirement=requirement.name,
                category=requirement.category,
                weight=requirement.weight,
                evidence=evidence_text,
            )
        )
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Could not assess requirement '{requirement.name}' with the model.") from exc
    return response


def compute_score(assessments: list[RequirementAssessment]) -> dict[str, Any]:
    scored = [a for a in assessments if a.status != "Needs Review"]
    total_weight = sum(a.weight for a in scored)
    if total_weight == 0:
        return {
            "overall_score": 0.0,
            "total_requirements": len(assessments),
            "met_count": sum(1 for a in assessments if a.status == "Met"),
            "partially_met_count": sum(1 for a in assessments if a.status == "Partially Met"),
            "not_evidenced_count": sum(1 for a in assessments if a.status == "Not Evidenced"),
            "needs_review_count": sum(1 for a in assessments if a.status == "Needs Review"),
            "status_counts": {"Met": sum(1 for a in assessments if a.status == "Met"), "Partially Met": sum(1 for a in assessments if a.status == "Partially Met"), "Not Evidenced": sum(1 for a in assessments if a.status == "Not Evidenced"), "Needs Review": sum(1 for a in assessments if a.status == "Needs Review")},
        }

    earned = 0.0
    for item in scored:
        if item.status == "Met":
            earned += item.weight * 1.0
        elif item.status == "Partially Met":
            earned += item.weight * 0.5
        elif item.status == "Not Evidenced":
            earned += 0.0
    overall = (earned / total_weight) * 100
    return {
        "overall_score": round(overall, 2),
        "total_requirements": len(assessments),
        "met_count": sum(1 for a in assessments if a.status == "Met"),
        "partially_met_count": sum(1 for a in assessments if a.status == "Partially Met"),
        "not_evidenced_count": sum(1 for a in assessments if a.status == "Not Evidenced"),
        "needs_review_count": sum(1 for a in assessments if a.status == "Needs Review"),
        "status_counts": {"Met": sum(1 for a in assessments if a.status == "Met"), "Partially Met": sum(1 for a in assessments if a.status == "Partially Met"), "Not Evidenced": sum(1 for a in assessments if a.status == "Not Evidenced"), "Needs Review": sum(1 for a in assessments if a.status == "Needs Review")},
    }


def generate_recommendations(assessments: list[RequirementAssessment]) -> list[str]:
    recommendations: list[str] = []
    for assessment in assessments:
        if assessment.status in {"Not Evidenced", "Partially Met"}:
            suggestion = assessment.improvement_suggestion.strip()
            if suggestion:
                recommendations.append(f"{assessment.requirement}: {suggestion}")
    if not recommendations:
        recommendations.append("Strong alignment: continue to emphasize your strongest evidence and keep examples concrete in future interviews.")
    return recommendations[:6]


def generate_interview_questions(assessments: list[RequirementAssessment], llm: ChatOpenAI, n_questions: int = 6) -> list[InterviewQuestion]:
    valid = [a for a in assessments if a.status in {"Met", "Partially Met"} or a.status == "Needs Review"]
    if not valid:
        return []
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Create 5 to 10 interview questions grounded in the assessment. Use realistic, honest, resume-grounded questions. "
                "Return only valid JSON with an array of objects containing: question, category, why_asked, sample_answer, follow_up. "
                "The sample_answer must be honest and grounded in the resume; if uncertain, label it as 'Example to adapt'."
            ),
            (
                "human",
                "Assessment:\n{assessment_json}\n\nCreate {n_questions} questions.",
            ),
        ]
    )
    assessment_json = json.dumps([a.dict() for a in valid], ensure_ascii=False)
    structured_llm = llm.with_structured_output(list[InterviewQuestion])
    try:
        response = structured_llm.invoke(prompt.format(assessment_json=assessment_json, n_questions=n_questions))
    except Exception:
        return []
    return response[:n_questions]


def analyze_resume_against_job(resume_text: str, job_text: str, qdrant_url: str, embeddings: OpenAIEmbeddings, llm: ChatOpenAI) -> dict[str, Any]:
    collection_name = get_collection_name()
    resume_chunks = make_documents_from_resume(resume_text, "resume.pdf")
    try:
        index_resume_chunks(resume_chunks, embeddings, qdrant_url, collection_name)
        vector_store = QdrantVectorStore(
            client=QdrantClient(url=qdrant_url, timeout=10),
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        requirements = extract_requirements(job_text, llm)
        assessed: list[RequirementAssessment] = []
        for requirement in requirements:
            evidence_docs = retrieve_evidence(vector_store, requirement.name, top_k=4)
            if not evidence_docs:
                assessed.append(
                    RequirementAssessment(
                        requirement=requirement.name,
                        category=requirement.category,
                        weight=requirement.weight,
                        status="Not Evidenced",
                        explanation="No relevant resume evidence was retrieved for this requirement.",
                        supporting_evidence=[],
                        page_references=[],
                        improvement_suggestion="Add a direct example that demonstrates this requirement in the resume.",
                    )
                )
                continue
            assessed_result = assess_requirement(requirement, evidence_docs, llm)
            assessed.append(assessed_result)
        score_data = compute_score(assessed)
        recommendations = generate_recommendations(assessed)
        interview_questions = generate_interview_questions(assessed, llm)
        return {
            "requirements": assessed,
            "score": score_data,
            "recommendations": recommendations,
            "interview_questions": interview_questions,
            "collection_name": collection_name,
        }
    finally:
        delete_collection(collection_name, qdrant_url)


def model_error_message(exc: Exception) -> str:
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None and len(messages) < 3:
        text = str(current).strip()
        if text:
            messages.append(text)
        current = current.__cause__ or current.__context__
    message = " | ".join(messages).lower()
    if "api key" in message or "authentication" in message or "unauthorized" in message:
        if "401" in message or "incorrect api key" in message or "invalid api key" in message:
            return (
                "The provider rejected the API key. The project .env was loaded, but the key may be "
                "expired, revoked, or not valid for this provider/base URL."
            )
        return "The provider rejected the authentication request. Check the provider key and BASE_URL in .env."
    if "rate limit" in message or "quota" in message or "too many requests" in message:
        return "OpenAI quota or rate limits were reached. Please wait a moment and try again."
    if "embedding" in message or "embeddings" in message:
        return (
            "The embedding request failed. Confirm that your provider supports embeddings, "
            "that OPENAI_BASE_URL points to its OpenAI-compatible /v1 endpoint, "
            "and that OPENAI_EMBEDDING_MODEL is a valid embedding model. "
            f"Provider details: {' | '.join(messages[:2])}"
        )
    if "network" in message or "connection" in message or "timed out" in message:
        if "qdrant" in message:
            return "Qdrant could not be reached or initialized. Start Docker and confirm the QDRANT_URL value."
        return "A network error occurred while contacting OpenAI or Qdrant. Please check connectivity and try again."
    if "qdrant" in message:
        return "Qdrant could not be reached or initialized. Start Docker and confirm the QDRANT_URL value."
    if "malformed" in message or "json" in message or "structured" in message:
        return "The model returned malformed output. Please try a different job description or refresh the analysis."
    return "An unexpected error occurred during the analysis. Please verify your inputs and settings."
