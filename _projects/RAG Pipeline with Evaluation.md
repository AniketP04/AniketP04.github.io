---
layout: page
title: Production RAG System with Hybrid Search & Evaluation Framework
description: A production-ready RAG system that combines semantic vector search (FAISS) with keyword matching (BM25) to deliver high-quality, citation-backed answers. Built with comprehensive evaluation metrics, monitoring, and one-command Docker deployment.
img: assets/img/rag_eval_proj.gif
importance: 1
related_publications: false
---

# Production RAG System with Hybrid Search & Evaluation Framework

Enterprise-grade Retrieval-Augmented Generation (RAG) system combining semantic vector search and keyword retrieval for accurate, citation-backed responses.

## Overview
This project implements a production-ready RAG pipeline using:
- FAISS vector search
- BM25 keyword search
- OpenAI LLM integration
- Automated evaluation framework
- Docker-based deployment

The system retrieves relevant context from ArXiv ML/AI papers and generates grounded answers with source citations.

## Key Features
- Hybrid retrieval (FAISS + BM25) + Reranking
- Citation-backed answer generation
- <2s query latency
- Automated evaluation metrics
- Docker deployment support
- FastAPI
- Monitoring and performance tracking

## Results
- Indexed **7,700+ documents**
- Achieved **94% faithfulness**
- Average latency: **~1.5s**
- Low-cost inference using GPT-4.1-mini
- Improved retrieval precision using hybrid search and reranking

## Tech Stack
- Python
- FastAPI
- FAISS
- BM25
- SentenceTransformers
- OpenAI API
- Docker
- PyTorch

## System Architecture
- Document ingestion & chunking
- Embedding generation
- Hybrid retrieval and reranking pipeline
- LLM-based response generation
- Evaluation & monitoring framework

## Applications
- Enterprise knowledge assistants
- AI-powered search engines
- Research paper QA systems
- Internal documentation retrieval
- Customer support automation

## Repository
[GitHub Repository](https://github.com/AniketP04/RAG-Pipeline-with-Evaluation)

