# JobAlign AI

JobAlign AI is a Streamlit-based application that compares a resume PDF against a job description and gives a structured match score, assessment by requirement, improvement recommendations, and interview-prep questions.

## Features

- Upload a resume as a PDF
- Paste a job description or job posting text
- Extract resume text using PyMuPDF
- Split resume content into chunks and store them in a temporary Qdrant collection
- Use OpenAI embeddings and LLMs to assess alignment against job requirements
- Produce an overall match score, per-requirement assessment, recommendations, and interview prompts
- Run locally with Dockerized Qdrant for vector storage

## Tech stack

- Python 3.11+
- Streamlit
- OpenAI
- LangChain
- Qdrant
- PyMuPDF

## Local setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   # On Windows PowerShell:
   # .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a local environment file:

   ```bash
   copy .env.example .env
   ```

4. Update `.env` with your provider API key and OpenAI-compatible base URL.

   ```env
   OPENAI_API_KEY=your_provider_api_key_here
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_CHAT_MODEL=gpt-4.1-mini
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
   QDRANT_URL=http://localhost:6333
   ```

5. Start Qdrant:

   ```bash
   docker compose up -d
   ```

6. Run the app:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the local Streamlit URL in your browser.
2. Upload a resume PDF.
3. Paste the job description.
4. Click **Analyze resume**.
5. Review the match score, requirement status, recommendations, and interview questions.

## Notes

- The app creates a temporary Qdrant collection for each analysis and deletes it afterward.
- The process expects a valid provider API key, provider-compatible chat and embedding models, and a running Qdrant service.
- For a third-party OpenAI-compatible provider, set `OPENAI_BASE_URL` to its `/v1` endpoint. `OPENAI_API_BASE` and `BASE_URL` are also accepted as compatibility aliases.
- Chat and embedding requests use the same `OPENAI_BASE_URL`. Set `OPENAI_EMBEDDING_MODEL` to the provider's exact embedding model name; not every chat provider supports embeddings.
- Unsupported or scanned PDFs may fail extraction if the file contains no readable text.

## Troubleshooting

- If Qdrant is unavailable, run `docker compose up -d` and verify the service is healthy.
- If the provider key is missing, update `.env` with `OPENAI_API_KEY` and `OPENAI_BASE_URL`, then restart the app.
- If the resume is a scanned image-only PDF, convert it to a searchable PDF first.
