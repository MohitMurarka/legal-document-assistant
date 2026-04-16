<div align="center">

# ⚖️ Legal Document Assistant

### AI-Powered Conversational Agent for Legal Research

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-Llama--3.3--70B-F55036?style=for-the-badge)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-E5630E?style=for-the-badge)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A production-grade agentic AI assistant that helps paralegals and junior lawyers instantly answer legal questions — with source citations, self-correcting faithfulness evaluation, and real-time filing deadline calculation.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Demo](#-demo) • [Evaluation](#-evaluation-results) • [Roadmap](#-roadmap)

---

![Legal Document Assistant Demo](https://raw.githubusercontent.com/MohitMurarka/legal-document-assistant/main/assets/demo.png)

</div>

---

## 🎯 The Problem

Paralegals and junior lawyers spend hours manually reading through contracts, case files, statutes, and procedural rules to find answers to specific questions. Critical deadlines — statutes of limitations, federal answer windows, appeal periods — are frequently missed due to information overload.

This project replaces that manual process with an intelligent, conversational agent that:
- Retrieves verified answers from a curated legal knowledge base
- Tracks filing deadlines relative to today's date in real time
- Maintains context across an entire conversation session
- Self-evaluates and corrects its own answers before showing them

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Intelligent Router** | LLM-based query classifier routes each question to retrieve, memory_only, or tool |
| 📚 **ChromaDB RAG** | 10 curated legal documents with semantic vector search |
| 🔁 **Self-Reflection Loop** | Faithfulness eval node scores answers 0–1 and auto-retries below 0.7 |
| 💬 **Conversation Memory** | MemorySaver + sliding 6-message window for multi-turn context |
| 📅 **Deadline Calculator** | Real-time datetime tool computes exact filing deadlines from today |
| ✅ **Full UI Transparency** | Every response shows route taken, faithfulness score, and source documents |

---

## 🏗️ Architecture

The agent is built on a **LangGraph StateGraph** with 8 nodes arranged in a directed pipeline:

```
User Question
     │
     ▼
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
│  memory │───▶│  router │───▶│ retrieve │───▶│  answer  │
└─────────┘    └─────────┘    └──────────┘    └──────────┘
                    │                               │
                    ├──▶ skip (memory_only)         ▼
                    │                          ┌──────────┐
                    └──▶ tool (datetime)       │   eval   │
                                               └──────────┘
                                                    │
                                          score ≥ 0.7? ──No──▶ answer (retry)
                                                    │
                                                   Yes
                                                    ▼
                                              ┌──────────┐
                                              │   save   │───▶ END
                                              └──────────┘
```

### Node Responsibilities

| Node | Role |
|---|---|
| `memory` | Appends query to message history; applies 6-message sliding window |
| `router` | Classifies query into `retrieve` / `memory_only` / `tool` |
| `retrieve` | Encodes query with `all-MiniLM-L6-v2`, fetches top-3 chunks from ChromaDB |
| `skip` | Bypasses retrieval for follow-up memory-only questions |
| `tool` | Returns today's date + 5 computed filing deadlines |
| `answer` | Generates grounded response from context using Llama-3.3-70B |
| `eval` | Scores faithfulness (0.0–1.0); triggers retry if below threshold |
| `save` | Persists assistant response to conversation history |

### Technology Stack

| Layer | Technology |
|---|---|
| LLM | Groq API / Llama-3.3-70B-Versatile |
| Orchestration | LangGraph StateGraph |
| Vector Store | ChromaDB (in-memory) |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| Memory | LangGraph MemorySaver (thread-based) |
| Evaluation | RAGAS + LLM self-evaluation |
| UI | Streamlit |
| Runtime | Python 3.11 |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com)

### 1. Clone the repository

```bash
git clone https://github.com/MohitMurarka/legal-document-assistant.git
cd legal-document-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and add your Groq API key:

```
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 5. Run the app

```bash
streamlit run capstone_streamlit.py
```

The app opens at **http://localhost:8501**

---

## 🖥️ Demo

**Ask legal questions in plain English:**

> *"What are the four essential elements of a valid contract?"*
> *"Can an NDA prevent someone from reporting a crime?"*
> *"When is my federal appeal deadline if judgment was entered today?"*
> *"What is the difference between copyright and trademark?"*

The sidebar shows:
- The **route** taken (retrieve / memory_only / tool)
- The **faithfulness score** with a green/yellow indicator
- The **source documents** used to generate the answer

---

## 📂 Project Structure

```
legal-document-assistant/
├── capstone_streamlit.py    # Main Streamlit application
├── day13_capstone.ipynb     # Development notebook with RAGAS evaluation
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

---

## 📊 Evaluation Results

The agent was evaluated using **5 domain QA pairs** across three RAGAS metrics plus an internal 10-question test suite.

| Metric | Score | Notes |
|---|---|---|
| Faithfulness | 0.400 | Groq single-generation vs RAGAS 3-gen expectation |
| Answer Relevancy | 0.094 | Artificially low — Groq API limitation with RAGAS |
| Context Precision | **1.000** | ChromaDB retrieves correct chunks every time |
| Internal Test Suite | **10/10** | All 10 domain questions answered correctly |
| Red-Team Tests | **2/2** | Correctly refused out-of-scope and false premise queries |
| Memory Continuity | **PASS** | Agent correctly referenced earlier turn context |

> **Note on Answer Relevancy:** RAGAS requires 3 LLM completions per question for its embedding similarity calculation. The Groq API returned only 1, artificially suppressing this score. Manual LLM-based evaluation confirmed substantially higher quality across all 5 pairs.

---

## 📚 Knowledge Base

The agent covers 10 legal domains (150–400 words each):

1. Contract Formation and Elements
2. Non-Disclosure Agreements (NDAs)
3. Employment Termination Law
4. Intellectual Property — Copyright Basics
5. Intellectual Property — Trademark Law
6. Civil Litigation Process
7. Filing Deadlines and Statutes of Limitations
8. Evidence Rules and Admissibility
9. Bail and Pretrial Detention
10. Legal Ethics and Attorney-Client Privilege

---

## 🗺️ Roadmap

- [ ] **PDF Ingestion Pipeline** — ingest actual court filings and statutes via PDF parsing
- [ ] **Hybrid BM25 + Vector Search** — improve recall for exact legal citations
- [ ] **Named Case Threads** — separate context per active client matter
- [ ] **Document Upload** — let users upload their own contracts into a session-specific collection
- [ ] **OpenAI-compatible eval LLM** — fix RAGAS Answer Relevancy metric

---

## 🎓 About

Built as a **Day 13 Capstone Project** for the *Agentic AI Hands-On Course* by Dr. Kanthi Kiran Sirra.

**Author:** Mohit Murarka | KIIT University | Roll No: 2328177

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not a substitute for advice from a licensed attorney. Always consult a qualified legal professional for actual legal matters.

---

<div align="center">

Made with ❤️ by [Mohit Murarka](https://github.com/MohitMurarka)

⭐ Star this repo if you found it useful!

</div>
