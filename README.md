# EARIA Agent

**An Economic Agent for Regulatory Impact Analysis in Colombian Telecommunications**

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Prerequisites](#prerequisites)  
4. [Installation](#installation)  
5. [Directory Structure](#directory-structure)  
6. [How It Works (High-Level Flow)](#how-it-works-high-level-flow)  
7. [Module Descriptions](#module-descriptions)  
8. [Configuration (.env)](#configuration-env)  
9. [Running the Agent](#running-the-agent)  
10. [Testing](#testing)  
11. [Extending or Customizing](#extending-or-customizing)  
12. [Troubleshooting](#troubleshooting)  
13. [Contributing](#contributing)  
14. [License](#license)  
15. [Contact](#contact)

---

## Project Overview

EARIA (Economic Agent for Regulatory Impact Analysis) is a retrieval-augmented generation (RAG) pipeline designed to analyze regulatory, economic, or legal documents related to Colombian telecommunications. It:

1. **Ingests** a directory of documents (PDFs, DOCXs, TXTs, HTML, etc.).  
2. **Cleans and Chunks** each document into smaller pieces to fit within LLM context windows.  
3. **Embeds & Indexes** those chunks in a local Chroma vector store using sentence-transformer embeddings.  
4. **Retrieves** the top-K relevant chunks in response to a user query.  
5. **Prompts** a local LLM (via Ollama) with curated templates (RAG, CoT, ToT, AIN extraction) and the retrieved context.  
6. **Generates** a structured, persona-driven economic-analysis response.

This repository provides a complete, ready-to-run codebase for local RAG: from document loading to final LLM call.

---

## Features

- **Multi-format Document Loader**:  
  - Supports PDF (`.pdf`), plain text (`.txt`), Word (`.docx`), HTML (`.html`/`.htm`), and generic files.  
  - Automatically tags each chunk with its source filename in metadata.

- **Configurable Text Cleaner & Chunker**:  
  - Normalizes whitespace, removes zero-width characters, and splits text into ~1000-character chunks (200 char overlap).  
  - Ensures LLM input stays under model limits and maintains context continuity.

- **Local Chroma Vector Store**:  
  - Uses a HuggingFace “all-MiniLM-L6-v2” embedding by default (configurable via environment).  
  - Stores chunk embeddings and metadata persistently for fast retrieval.

- **LLM Abstraction via Ollama**:  
  - Wraps `ChatOllama` (default: `mistral:7b-instruct` at `http://localhost:11434`).  
  - Offers four prompt templates:
  1. **Basic RAG**  
  2. **Chain-of-Thought (CoT)**  
  3. **Tree-of-Thought (ToT)**  
  4. **AIN extraction** (Extracts “Actor–Impact–Needs” components)

- **Modular “EARIAAgent” Class**:  
  - Drives the end-to-end pipeline: load documents, chunk, index, retrieve, prompt, and return.  
  - Easily swapped or extended (e.g., change vector store, replace prompt templates).

- **Test Suite**:  
  - Validates basic functionality of document loading and agent initialization.  

- **CLI Entry Point**:  
  - Simple command:  
    ```bash
    python main.py --docs <PATH_TO_DOCS> --query "<YOUR_QUERY>"
    ```
  - Prints LLM’s analysis to `stdout`.

---

## Prerequisites

1. **Python 3.9+**  
2. **Ollama** (for running a local chat model)  
   - Must have a model like `mistral:7b-instruct` installed and Ollama daemon listening (default: `http://localhost:11434`).  
3. **pip (or pipenv, poetry) for Python dependencies**.  
4. **(Optional) Git** to clone the repository.
