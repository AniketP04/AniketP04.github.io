---
layout: page
title: Production RAG System with Hybrid Search & Evaluation Framework
description: Production-ready RAG system using FAISS, BM25, Docker, and evaluation metrics.
img: assets/img/rag_eval_proj.gif
importance: 1
related_publications: false
github: https://github.com/AniketP04/RAG-Pipeline-with-Evaluation
---

A production-ready RAG system that combines semantic vector search (FAISS) with keyword matching (BM25) to deliver high-quality, citation-backed answers. Built with comprehensive evaluation metrics, monitoring, and one-command Docker deployment.

# Production RAG System with Hybrid Search & Evaluation Framework

Enterprise-grade Retrieval-Augmented Generation (RAG) system combining semantic vector search and keyword retrieval for accurate, citation-backed responses.

<img width="2451" height="2275" alt="image" src="https://github.com/user-attachments/assets/d4738aaf-e154-4e0c-a503-466b9d5a4bf0" />

## Overview
This project implements a production-ready RAG pipeline using:
- FAISS vector search
- BM25 keyword search
- OpenAI LLM integration
- Automated evaluation framework
- Docker-based deployment

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/arxiv.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The system retrieves relevant context from [ArXiv ML/AI papers](https://huggingface.co/datasets/ccdv/arxiv-summarization) and generates grounded answers with source citations.

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

## Applications
- Enterprise knowledge assistants
- AI-powered search engines
- Research paper QA systems
- Internal documentation retrieval
- Customer support automation
