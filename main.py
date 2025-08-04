# main.py (Single Endpoint with Auto-Cancellation)
import os
import json
import uuid
import redis
import asyncio # Import asyncio
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# --- 1. SETUP & CONSTANTS ---
load_dotenv()
EXPECTED_AUTH_TOKEN = "c88d7e70b6c77cd88271a48126bcd54761315985a275d864cd7e2b7ba342f1cf"
redis_conn = redis.from_url(os.getenv("REDIS_URL"))

# --- 2. FASTAPI APP DEFINITION ---
app = FastAPI(title="Intelligent Query-Retrieval System")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=AnswerResponse)
async def process_queries_and_wait(request: QueryRequest, http_request: Request):
    # --- A. Authentication ---
    auth_header = http_request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    if token != EXPECTED_AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

    # --- B. Job Creation ---
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "document_url": request.documents,
        "questions": request.questions
    }
    
    # --- C. Push job to Redis queue ---
    redis_conn.lpush('job_queue', json.dumps(job_data))
    print(f"Job {job_id} created for document: {request.documents}")

    # --- D. Wait for Result OR Client Disconnect ---
    result_key = f"result:{job_id}"
    cancel_key = f"cancel:{job_id}"
    
    try:
        while True:
            # Check if the client has disconnected
            if await http_request.is_disconnected():
                print(f"Client disconnected for job {job_id}. Sending cancellation signal.")
                # Set the cancel flag for the worker
                redis_conn.set(cancel_key, "1", ex=600)
                # Stop waiting and raise an exception on the server side.
                raise HTTPException(status_code=499, detail="Client closed request.")

            # Check if the result is ready (non-blocking check)
            result_json = redis_conn.rpop(result_key)
            if result_json:
                result_data = json.loads(result_json)
                print(f"Result found for job {job_id}. Returning to client.")
                return result_data

            # Wait for 1 second before checking again
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        # This block handles the server shutting down or other cancellations
        print(f"Request loop cancelled for job {job_id}. Sending cancellation signal.")
        redis_conn.set(cancel_key, "1", ex=600)
        raise
