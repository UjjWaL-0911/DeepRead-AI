# DeepRead-AI: Hybrid Search RAG Pipeline

**Authors:** Ujjwal Krishna Khanna & Pratham Kala *(Equal Contribution)* **Context:** Bachelor Thesis Project (BTP) — Document AI

## Overview
DeepRead-AI is an advanced Retrieval-Augmented Generation (RAG) backend designed for high-performance document intelligence and layout parsing. It ingests diverse document formats (PDF, DOCX, PPTX, XLSX, Images, Web Pages), indexes their contents using a dual-strategy approach (Semantic + Keyword), and provides highly accurate, context-aware answers to user queries. 

The architecture features a unified interface where a minimalistic, Tailwind-styled frontend interacts seamlessly with the FastAPI backend, removing the need for local tunneling tools like ngrok.

## Architecture & Tech Stack
* **Backend Framework:** FastAPI
* **Embedding Model:** OpenAI `text-embedding-3-small`
* **Vector & Keyword Data Store:** DataStax AstraDB
* **Reranker:** Cross-Encoder (`BAAI/bge-reranker-base`)
* **LLM:** OpenAI `gpt-4o-mini`
* **Caching & State:** Redis
* **Frontend:** HTML/JS with TailwindCSS

## Key Features
* **Hybrid Retrieval:** Combines dense vector search (semantic) with an inverted index lookup (keyword) to maximize context retrieval accuracy.
* **Dynamic Reranking:** Applies a weighted Cross-Encoder reranking step to evaluate and elevate the most relevant chunks before passing them to the LLM.
* **Multimodal Parsing:** Capable of parsing standard text documents and extracting text from images using Vision APIs.
* **Automated Translation:** Detects foreign languages and automatically translates document context into English prior to indexing.

## Deployment Guide (Render.com)
This application is structured for a seamless, production-ready deployment on Render.

1. **Repository Setup:** Ensure `main.py`, `worker.py`, and `requirements.txt` are in the root directory. **Do not** commit your `.env` file.
2. **Render Web Service:** Create a new Web Service on Render and connect this GitHub repository.
3. **Configuration:**
   * **Language:** Python 3
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables:** Add the following keys in the Render dashboard under your service settings:
   * `OPENAI_API_KEY`
   * `ASTRA_DB_API_ENDPOINT`
   * `ASTRA_DB_APPLICATION_TOKEN`
   * `REDIS_URL`

Once deployed, Render will provide a live URL where you can access the frontend and interact with the RAG pipeline directly.
