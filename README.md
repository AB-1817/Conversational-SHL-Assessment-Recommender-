# Conversational SHL Assessment Recommender

A conversational recommendation service that helps hiring managers and recruiters find the most suitable assessments from the SHL product catalog through multi-turn dialogue.

The service is built as a stateless FastAPI application that leverages semantic search (FAISS) and LLMs (Groq Llama 3) with robust guardrails, RAG context, and strict schema validation.

---

## Key Features

*   **Stateless Chat Endpoint (`POST /chat`)**: Processes the entire conversation history on every request, keeping the backend lightweight and database-free.
*   **Behavioral Intelligence**:
    *   **Clarification**: Asks targeted clarifying questions if the hiring context is vague.
    *   **Recommendation**: Recommends a tailored battery of 1–10 assessments once context is established.
    *   **Refinement**: Dynamically updates the recommended shortlist mid-conversation based on updated requirements.
    *   **Comparison**: Compares assessments using grounded catalog details without offering empty list recommendations on comparison turns.
*   **Scope Guardrails**: Detects and politely declines legal compliance questions, general HR advice, or prompt injection attempts, while preserving existing shortlist recommendations.
*   **No Hallucinations**: Validates and grounds all LLM-generated URLs against the local product catalog before responding.
*   **Turn-Limit Safeguard**: Tracks turns and automatically forces the final shortlist lock-in by turn 7 or 8.

---

## Tech Stack

*   **Backend Framework**: FastAPI & Pydantic (strict schema enforcement)
*   **Vector DB & Retrieval**: FAISS (IndexFlatIP) and FastEmbed (`BAAI/bge-small-en-v1.5`)
*   **Inference Engine**: Groq Cloud SDK (`llama-3.3-70b-versatile`)
*   **Testing**: Pytest & HTTPX

---

## Local Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Clone the repository and navigate into the root directory.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install pytest
```

### 3. Setup Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Build the Vector Search Index
The vector database is built using the provided product catalog dataset (`data/catalog.json`). Run the indexer script to generate the FAISS embeddings:
```bash
python -m indexer.build_index
```
This downloads the ONNX embedding model locally and generates the index under `data/faiss_index/`.

### 5. Start the FastAPI Server
Start the development server with auto-reload:
```bash
python -m uvicorn main:app --reload
```
The API documentation will be available locally at `http://127.0.0.1:8000/docs`.

---

## Running Tests

An integration test suite is included to verify schema compliance and conversational edge-cases. Run the tests using:
```bash
python -m pytest tests/ -v
```

---

## Deployment

A production-ready `Dockerfile` is included. To deploy to hosting services like Render:

1. Create a new **Web Service** on Render connected to your repository.
2. Select **Docker** as the environment type.
3. Add the environment variable:
   *   `GROQ_API_KEY` = `<your-api-key>`
4. Deploy. Render will automatically build the container and spin up the service.
