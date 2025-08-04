import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import requests
import tempfile
import fitz
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
import time

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

# Tuning params (adjust for your OpenAI limits)
MAX_BATCH_SIZE = 2048      # Up to 2,048 input texts/string per request (if total <= 8,191 tokens)
MAX_CONCURRENT_EMBS = 8    # Number of parallel embedding requests (adjust if you hit OpenAI rate limits)
UPSERT_BATCH_SIZE = 1000   # Pinecone max upsert batch size (usually 1k)
MAX_CONCURRENT_UPSERTS = 4 # Pinecone upsert concurrency

def logtime(phase, start, last=None):
    now = time.perf_counter()
    since_start = now - start
    since_last = (now - last) if last else None
    if since_last:
        print(f"⏱️ [{phase}] {since_start:.2f}s (+{since_last:.2f}s since last)")
    else:
        print(f"⏱️ [{phase}] {since_start:.2f}s")
    return now

async def async_get_embeddings(batch: List[str]) -> List[List[float]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: [e.embedding for e in sync_openai_client.embeddings.create(input=[t.replace('\n', ' ') for t in batch], model=EMBEDDING_MODEL_API).data]
    )

async def batch_embeddings_concurrent(texts: List[str]) -> List[List[float]]:
    # Use up to max-sized batches and parallelism for OpenAI
    batches = [texts[i:i+MAX_BATCH_SIZE] for i in range(0, len(texts), MAX_BATCH_SIZE)]
    sem = asyncio.Semaphore(MAX_CONCURRENT_EMBS)
    results = []

    async def embed_and_acquire(batch):
        async with sem:
            return await async_get_embeddings(batch)
    coros = [embed_and_acquire(batch) for batch in batches]
    for coro in asyncio.as_completed(coros):
        results.extend(await coro)
    return results

async def async_upsert(batch):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: index.upsert(vectors=batch)
    )

async def batch_upsert_concurrent(vectors: List[dict]):
    batches = [vectors[i:i+UPSERT_BATCH_SIZE] for i in range(0, len(vectors), UPSERT_BATCH_SIZE)]
    sem = asyncio.Semaphore(MAX_CONCURRENT_UPSERTS)
    async def upsert_and_acquire(batch):
        async with sem:
            return await async_upsert(batch)
    coros = [upsert_and_acquire(batch) for batch in batches]
    await asyncio.gather(*coros)

def process_pdf_chunks(file_path: str) -> List[str]:
    doc = fitz.open(file_path)
    full_text = "".join(page.get_text() for page in doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    return text_splitter.split_text(full_text)

async def process_and_index_pdf(file_path: str, document_id: str, cancel_key: str):
    t0 = time.perf_counter()
    print(f"⚙️ Starting Hybrid indexing for: {document_id}")
    chunks = process_pdf_chunks(file_path)
    t1 = logtime("Loaded & split PDF", t0)

    bm25 = BM25Encoder()
    bm25.fit(chunks)
    t2 = logtime("BM25 fit", t0, t1)

    dense_embeddings = await batch_embeddings_concurrent(chunks)
    t3 = logtime("All embeddings", t0, t2)

    sparse_vectors = bm25.encode_documents(chunks)
    t4 = logtime("BM25 sparse", t0, t3)

    vectors_to_upsert = [
        {
            "id": f"{document_id}-chunk-{i}",
            "values": dense,
            "sparse_values": sparse,
            "metadata": {"text": chunk, "document_id": document_id}
        } for i, (chunk, dense, sparse) in enumerate(zip(chunks, dense_embeddings, sparse_vectors))
    ]
    await batch_upsert_concurrent(vectors_to_upsert)
    t5 = logtime("Pinecone upsert", t0, t4)
    print(f"✅ Finished hybrid indexing in {t5-t0:.2f}s with {len(chunks)} chunks.")

def get_embeddings_sync(texts: List[str]) -> List[List[float]]:
    return [e.embedding for e in sync_openai_client.embeddings.create(input=[t.replace('\n', ' ') for t in texts], model=EMBEDDING_MODEL_API).data]

def retrieve_and_rerank_context(original_query: str, document_id: str, k_retrieve=10, k_rerank=5) -> List[str]:
    start = time.perf_counter()
    dense_embedding = get_embeddings_sync([original_query])[0]
    t1 = logtime("Embedding query", start)
    bm25 = BM25Encoder()
    bm25.fit([original_query])
    sparse_embedding = bm25.encode_queries(original_query)
    t2 = logtime("BM25 encode query", start, t1)

    results = index.query(
        vector=dense_embedding,
        sparse_vector=sparse_embedding,
        alpha=0.5,
        top_k=k_retrieve,
        include_metadata=True,
        filter={"document_id": {"$eq": document_id}}
    )
    t3 = logtime("Pinecone query", start, t2)
    candidate_chunks = [m['metadata']['text'] for m in results['matches']]
    if not candidate_chunks:
        return []
    rerank_pairs = [[original_query, chunk] for chunk in candidate_chunks]
    scores = reranker_model.predict(rerank_pairs)
    t4 = logtime("Reranking", start, t3)
    scored_chunks = sorted(zip(scores, candidate_chunks), key=lambda x: x[0], reverse=True)
    result = [chunk for score, chunk in scored_chunks[:k_rerank]]
    print(f"⏱️ Retrieval+Rerank total: {time.perf_counter() - start:.2f}s")
    return result

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
        start = time.perf_counter()
        response = await openrouter_client.chat.completions.create(
            model="mistralai/mistral-7b-instruct-v0.2",
            messages=[{"role": "user", "content": prompt}]
        )
        print(f"⏱️ LLM answer for question in {time.perf_counter() - start:.2f}s")
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
                    await process_and_index_pdf(temp_pdf_path, document_id, cancel_key)
                    redis_conn.sadd(PROCESSED_DOCS_SET_KEY, document_id)
                print(f"Processing {len(questions)} questions in a batch for job {job_id}...")
                answer_generation_tasks = []
                for question in questions:
                    if redis_conn.exists(cancel_key): raise InterruptedError("Job canceled")
                    context = retrieve_and_rerank_context(question, document_id)
                    answer_generation_tasks.append((question, context))
                batch_start = time.perf_counter()
                all_answers = await get_llm_answers_in_batch(answer_generation_tasks)
                print(f"⏱️ LLM answers for batch in {time.perf_counter() - batch_start:.2f}s")
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
