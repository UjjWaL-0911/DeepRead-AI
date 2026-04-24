# --- IMPORTS ---
import os
import tempfile
import requests
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
from urllib.parse import urlparse

# --- WORKER IMPORTS ---
from worker import (
    load_environment_config,
    initialize_services,
    generate_document_id,
    process_and_index_pdf,
    process_and_index_docx,
    process_and_index_url,
    process_and_index_pptx,
    process_and_index_excel,
    process_and_index_image,
    answer_queries,
    PROCESSED_DOCS_KEY,
    RAG_CONFIG
)

# --- LIFESPAN MANAGEMENT & GLOBAL SERVICES ---
services: Dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 API starting up...")
    config = load_environment_config()
    services.update(initialize_services(config))
    print("✅ Services initialized. API is ready.")
    yield
    print("👋 API shutting down...")
    services.clear()

app = FastAPI(
    title="DeepRead AI RAG API",
    description="Backend and Minimal Frontend for RAG Pipeline.",
    version="1.1.0",
    lifespan=lifespan
)

# Allow CORS if needed later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class JobRequest(BaseModel):
    documents: str = Field(..., description="A URL pointing to the document.")
    questions: List[str] = Field(..., description="List of questions.")
    semantic_weight: float = Field(RAG_CONFIG["semantic_weight"])
    keyword_weight: float = Field(RAG_CONFIG["keyword_weight"])
    k_semantic: int = Field(RAG_CONFIG["k_semantic"])
    k_keyword: int = Field(RAG_CONFIG["k_keyword"])
    k_rerank: int = Field(RAG_CONFIG["k_rerank"])

class JobResponse(BaseModel):
    job_id: str = Field(...)
    answers: List[str] = Field(...)

# --- MINIMALISTIC FRONTEND ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepRead AI - Document Q&A</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 text-gray-800 font-sans min-h-screen p-8">
    <div class="max-w-3xl mx-auto bg-white p-8 rounded-lg shadow-md border border-gray-200">
        <h1 class="text-3xl font-bold text-blue-600 mb-2">DeepRead AI</h1>
        <p class="text-gray-500 mb-6">Upload a document URL and ask questions to the RAG pipeline.</p>
        
        <form id="ragForm" class="space-y-4">
            <div>
                <label class="block text-sm font-medium text-gray-700">Document URL</label>
                <input type="url" id="docUrl" required placeholder="https://example.com/document.pdf" 
                       class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 p-2 border">
            </div>
            
            <div>
                <label class="block text-sm font-medium text-gray-700">Questions (one per line)</label>
                <textarea id="questions" required rows="4" placeholder="What is the main topic?\nWho are the authors?" 
                          class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 p-2 border"></textarea>
            </div>
            
            <button type="submit" id="submitBtn" class="w-full bg-blue-600 text-white font-bold py-2 px-4 rounded hover:bg-blue-700 transition">
                Process & Answer
            </button>
        </form>

        <div id="loading" class="hidden mt-6 text-center text-blue-600 font-semibold animate-pulse">
            Processing document and generating answers. This may take a moment...
        </div>

        <div id="results" class="hidden mt-8 space-y-4 border-t pt-6">
            <h2 class="text-xl font-bold">Results</h2>
            <div id="answersContainer" class="space-y-4"></div>
        </div>
    </div>

    <script>
        document.getElementById('ragForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const container = document.getElementById('answersContainer');
            
            btn.disabled = true;
            btn.classList.add('opacity-50');
            loading.classList.remove('hidden');
            results.classList.add('hidden');
            container.innerHTML = '';

            const url = document.getElementById('docUrl').value;
            const questions = document.getElementById('questions').value.split('\\n').filter(q => q.trim() !== '');

            try {
                const response = await fetch('/hackrx/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        documents: url, 
                        questions: questions,
                        semantic_weight: 0.7, keyword_weight: 0.3, k_semantic: 30, k_keyword: 30, k_rerank: 15
                    })
                });

                const data = await response.json();
                
                if (!response.ok) throw new Error(data.detail || 'Something went wrong');

                data.answers.forEach((ans, idx) => {
                    container.innerHTML += `
                        <div class="bg-gray-50 p-4 rounded border">
                            <p class="font-semibold text-sm text-gray-600">Q: ${questions[idx]}</p>
                            <p class="mt-2 text-gray-800">A: ${ans}</p>
                        </div>
                    `;
                });
                results.classList.remove('hidden');
            } catch (error) {
                container.innerHTML = `<div class="bg-red-50 text-red-600 p-4 rounded border border-red-200">Error: ${error.message}</div>`;
                results.classList.remove('hidden');
            } finally {
                btn.disabled = false;
                btn.classList.remove('opacity-50');
                loading.classList.add('hidden');
            }
        });
    </script>
</body>
</html>
"""

@app.get('/', response_class=HTMLResponse)
async def root():
    """Serves the minimalistic frontend UI."""
    return HTML_TEMPLATE

# --- CORE ENDPOINT ---
@app.post('/hackrx/run', response_model=JobResponse)
async def process_job(request: JobRequest):
    # YOUR EXACT ORIGINAL LOGIC GOES HERE (Truncated for brevity in reading, paste your code from here down)
    document_url = request.documents
    document_id = generate_document_id(document_url)

    if not services["redis_conn"].sismember(PROCESSED_DOCS_KEY, document_id): 
        temp_file_path = None
        try:
            url_path = urlparse(document_url).path
            file_extension = os.path.splitext(url_path)[-1].lower()

            if file_extension in [".pdf", ".docx",".pptx",".xlsx", ".xls",".jpg",".jpeg",".png"]:
                response = requests.get(document_url, timeout=30)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name
                if file_extension == ".pdf": await process_and_index_pdf(temp_file_path, document_id, services)
                elif file_extension == ".docx": await process_and_index_docx(temp_file_path, document_id, services)
                elif file_extension == ".pptx": await process_and_index_pptx(temp_file_path, document_id, services)
                elif file_extension in ['.xlsx', '.xls']: await process_and_index_excel(temp_file_path, document_id, services)
                elif file_extension in ['.jpg', '.jpeg', '.png']: await process_and_index_image(temp_file_path, document_id, services)
            else:
                await process_and_index_url(document_url, document_id, services)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
        finally:
            if temp_file_path: os.unlink(temp_file_path)

    try:
        answers = await answer_queries(
            queries=[(q, document_id) for q in request.questions],
            services=services,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight,
            k_semantic=request.k_semantic,
            k_keyword=request.k_keyword,
            k_rerank=request.k_rerank
        )
        return {"job_id": document_id, "answers": answers}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate answers: {str(e)}")