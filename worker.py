import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import requests
import tempfile
import fitz  # PyMuPDF
import pinecone
import redis
import openai
import asyncio
import nltk
from dotenv import load_dotenv
from typing import List, Tuple
from pinecone_text.sparse import BM25Encoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

# Download nltk resources at startup if needed
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

print("🚀 Worker starting up...")
load_dotenv()

redis_conn = redis.from_url(os.getenv("REDIS_URL"))
sync_openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "hackrx-hybrid-index-v1"
EMBEDDING_MODEL_API = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
PROCESSED_DOCS_SET_KEY = "processed_docs_hybrid_v1"

print("📦 Loading Cross-Encoder model for re-ranking (CPU)...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Re-ranker model loaded.")

print("🌲 Connecting to Pinecone index...")
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' not found. Creating it...")
    pc.create_index(
        name=INDEX_NAME, dimension=EMBEDDING_DIMENSION, metric='dotproduct',
        spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)
print("✅ Worker is ready and waiting for jobs.")
print("-" * 50)

def get_embeddings(texts: List[str], batch_size: int = 96) -> List[List[float]]:
    """Batched embedding generation for speed (OpenAI API limit for small model is 96 per batch)."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [t.replace("\n", " ") for t in texts[i:i+batch_size]]
        response = sync_openai_client.embeddings.create(input=batch, model=EMBEDDING_MODEL_API)
        for embedding in response.data:
            embeddings.append(embedding.embedding)
    return embeddings

def process_and_index_pdf(file_path: str, document_id: str, cancel_key: str):
    print(f"⚙️ Starting Hybrid Search indexing for: {document_id}")
    doc = fitz.open(file_path)
    full_text = "".join(page.get_text() for page in doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)
    print(f"📄 Document split into {len(chunks)} chunks.")

    bm25 = BM25Encoder()
    bm25.fit(chunks)
    batch_size = 1000  # Pinecone free tier max per upsert

    for i in range(0, len(chunks), batch_size):
        if redis_conn.exists(cancel_key):
            raise InterruptedError(f"Job {document_id} canceled during indexing.")
        batch_chunks = chunks[i:i + batch_size]
        dense_embeddings = get_embeddings(batch_chunks, batch_size=96)
        sparse_vectors = bm25.encode_documents(batch_chunks)
        vectors_to_upsert = [
            {
                "id": f"{document_id}-chunk-{i+j}",
                "values": dense_vec,
                "sparse_values": sparse_vec,
                "metadata": {"text": chunk_text, "document_id": document_id}
            } for j, (chunk_text, dense_vec, sparse_vec) in enumerate(zip(batch_chunks, dense_embeddings, sparse_vectors))
        ]
        print(f"📤 Upserting batch of {len(vectors_to_upsert)} hybrid vectors...")
        index.upsert(vectors=vectors_to_upsert)
    print(f"✅ Finished hybrid indexing with {len(chunks)} chunks.")

def retrieve_and_rerank_context(original_query: str, document_id: str, bm25_encoder: BM25Encoder = None, k_retrieve=20, k_rerank=5) -> List[str]:
    dense_embedding = get_embeddings([original_query])[0]
    bm25 = bm25_encoder or BM25Encoder()
    bm25.fit([original_query])  # fast; only fitting on one query, not slow
    sparse_embedding = bm25.encode_queries(original_query)
    results = index.query(
        vector=dense_embedding,
        sparse_vector=sparse_embedding,
        alpha=0.5,
        top_k=k_retrieve,
        include_metadata=True,
        filter={"document_id": {"$eq": document_id}}
    )
    candidate_chunks = [m['metadata']['text'] for m in results['matches']]
    if not candidate_chunks:
        return []
    rerank_pairs = [[original_query, chunk] for chunk in candidate_chunks]
    scores = reranker_model.predict(rerank_pairs)
    scored_chunks = sorted(zip(scores, candidate_chunks), key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks[:k_rerank]]

async def get_llm_answer_async(query: str, context_chunks: List[str]) -> str:
    if not context_chunks:
        return "Could not find relevant information."
    openrouter_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://github.com/UjjwaL-0911/DeepRead-AI.git",
            "X-Title": "Intelligent Query-Retrieval System"
        }
    )
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a precise assistant. Your answer MUST be based SOLELY on the CONTEXT provided. If the context does not contain the answer, you MUST state that you cannot answer. Provide your final answer as a single, clean paragraph.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:
"""
    try:
        response = await openrouter_client.chat.completions.create(
            model="mistralai/mistral-7b-instruct-v0.2",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting LLM answer for query '{query}': {e}")
        return "An error occurred while generating the answer."

async def get_llm_answers_in_batch(tasks: List[Tuple[str, List[str]]]) -> List[str]:
    async_tasks = [get_llm_answer_async(query, context) for query, context in tasks]
    all_answers = await asyncio.gather(*async_tasks)
    return all_answers

async def main_worker_loop():
    print("✅ Worker is ready. Press Ctrl+C to shut down.")
    try:
        while True:
            job_id, temp_pdf_path = None, None
            try:
                _, job_json = await asyncio.to_thread(redis_conn.brpop, 'job_queue')
                job_data = json.loads(job_json)
                job_id = job_data["job_id"]
                cancel_key = f"cancel:{job_id}"
                if redis_conn.exists(cancel_key):
                    print(f"⏩ Job {job_id} was canceled before starting.")
                    continue
                print(f"Processing job: {job_id}")
                document_url, document_id = job_data["document_url"], os.path.basename(job_data["document_url"].split('?')[0])
                questions = job_data["questions"]
                if not redis_conn.sismember(PROCESSED_DOCS_SET_KEY, document_id):
                    response = requests.get(document_url)
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                        temp_pdf.write(response.content)
                        temp_pdf_path = temp_pdf.name
                    if redis_conn.exists(cancel_key): raise InterruptedError("Job canceled")
                    process_and_index_pdf(temp_pdf_path, document_id, cancel_key)
                    redis_conn.sadd(PROCESSED_DOCS_SET_KEY, document_id)
                print(f"Processing {len(questions)} questions in a batch for job {job_id}...")
                answer_generation_tasks = []
                for question in questions:
                    if redis_conn.exists(cancel_key): raise InterruptedError("Job canceled")
                    context = retrieve_and_rerank_context(question, document_id)
                    answer_generation_tasks.append((question, context))
                all_answers = await get_llm_answers_in_batch(answer_generation_tasks)
                result = {"answers": all_answers}
                redis_conn.lpush(f"result:{job_id}", json.dumps(result))
                redis_conn.expire(f"result:{job_id}", 3600)
                print(f"✅ Finished job: {job_id}")
            except InterruptedError:
                print(f"🛑 Job {job_id} canceled by client.")
            except Exception as e:
                print(f"❌ Error on job {job_id}: {e}")
                if job_id:
                    err_res = {"answers": [f"An error occurred: {e}"]}
                    redis_conn.lpush(f"result:{job_id}", json.dumps(err_res))
            finally:
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    os.unlink(temp_pdf_path)
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C received. Shutting down worker...")
    finally:
        print("🔌 Closing connections.")
        redis_conn.close()

if __name__ == "__main__":
    asyncio.run(main_worker_loop())
