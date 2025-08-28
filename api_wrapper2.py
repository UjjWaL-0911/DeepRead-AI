# api_wrapper2.py

# --- IMPORTS ---
import os
import tempfile
import requests
import traceback
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from urllib.parse import urlparse

# --- WORKER IMPORTS ---
# Ensure this matches your worker file's name (e.g., worker2.0.py)
from worker2 import (
    load_environment_config,
    initialize_services,
    generate_document_id,
    process_and_index_pdf,
    process_and_index_docx,
    process_and_index_url,
    process_and_index_pptx,
    process_and_index_excel,
    process_and_index_image,
    discover_concepts_dynamically_async,
    extract_all_facts_from_document,
    analyze_facts_for_discrepancies,
    PROCESSED_DOCS_KEY
)

# --- LIFESPAN MANAGEMENT & GLOBAL SERVICES ---
# This dictionary will hold our initialized services (clients, models, etc.)
services: Dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes all necessary services on startup.
    """
    print("🚀 AI Deal Checker API starting up...")
    print("Loading environment configuration...")
    config = load_environment_config()
    print("Initializing all services (DB, AI Models, etc.)...")
    services.update(initialize_services(config))
    print("✅ Services initialized. API is ready.")
    yield
    print("👋 API shutting down...")
    services.clear()

app = FastAPI(
    title="AI Deal Checker API",
    description="API for autonomously finding discrepancies in financial documents.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models (New for Analysis Workflow) ---

class AnalysisRequest(BaseModel):
    document_url: str = Field(..., description="A URL pointing to the document (PDF, DOCX, etc.) to be analyzed.")

class FindingEvidence(BaseModel):
    value: str
    source_text: str

class Finding(BaseModel):
    type: str = Field(..., description="The type of discrepancy found (e.g., 'Data Inconsistency').")
    severity: str = Field(..., description="The severity of the finding (e.g., 'High', 'Medium').")
    message: str = Field(..., description="A human-readable message describing the issue.")
    evidence: List[FindingEvidence] = Field(..., description="A list of the conflicting values and their source text.")

class AnalysisResponse(BaseModel):
    document_id: str = Field(..., description="The unique identifier for the processed document.")
    status: str = Field(..., description="The status of the analysis job.")
    findings: List[Finding] = Field(..., description="A list of discrepancies found in the document.")

# --- API Endpoints ---

@app.post('/analyze/document', response_model=AnalysisResponse)
async def analyze_document(request: AnalysisRequest):
    """
    Accepts a document URL, processes it, and returns autonomously discovered discrepancies.
    """
    document_url = request.document_url
    document_id = generate_document_id(document_url)

    # --- STAGE 1: DOCUMENT PROCESSING & INDEXING ---
    if not services["redis_conn"].sismember(PROCESSED_DOCS_KEY, document_id):
        print(f"Document ID '{document_id}' not in cache. Starting ingestion pipeline...")
        temp_file_path = None
        try:
            url_path = urlparse(document_url).path
            file_extension = os.path.splitext(url_path)[-1].lower()

            if file_extension in [".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".jpg", ".jpeg", ".png"]:
                print(f"✅ Detected '{file_extension}' from URL. Downloading file...")
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
                print(f"⚠️ No specific file extension detected. Assuming it is a webpage...")
                await process_and_index_url(document_url, document_id, services)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
        finally:
            if temp_file_path: os.unlink(temp_file_path)
    else:
        print(f"✅ Document ID '{document_id}' found in cache. Skipping indexing.")

    # --- STAGE 2: AUTONOMOUS ANALYSIS ---
    try:
        print(f"🤖 Starting autonomous analysis for document ID: {document_id}")
        
        # --- CHANGE: Added Phase 0 to dynamically discover concepts from the document.
        concepts_to_track = await discover_concepts_dynamically_async(document_id, services)
        
        # --- CHANGE: Pass the dynamic 'concepts_to_track' into the extraction function.
        facts = await extract_all_facts_from_document(document_id, services, concepts_to_track)
        
        # Phase 2: Analyze the structured facts to find discrepancies
        findings = analyze_facts_for_discrepancies(facts)
        
        print("✅ Analysis pipeline complete.")
        return {"document_id": document_id, "status": "Completed", "findings": findings}
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during analysis phase: {str(e)}")

@app.get('/')
def root():
    """A simple endpoint to confirm the API is live."""
    return {"message": "AI Deal Checker API is live"}