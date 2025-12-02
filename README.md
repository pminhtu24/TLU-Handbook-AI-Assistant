# Handbook-AI-Assistant (ChatChitAI)

A student-handbook retrieval-augmented chatbot for Thuy Loi University — enables querying DOCX/PDF handbook materials via Vietnamese-language embeddings and local LLMs, with a Streamlit UI. Ask about rules, fees, exam schedules, GPA policies, dorm regulations, and all the usual handbook mysteries. No more scrolling through endless PDF pages.

---

## 🚀 What This Project Does

- Converts student handbook documents (DOCX, PDF) into clean Markdown for processing.  
- Indexes content using hybrid retrieval (Milvus + BM25).  
- Embeds Vietnamese text using pretrained Vietnamese embeddings.  
- Provides a RAG (Retrieval-Augmented Generation) chatbot interface that accepts Vietnamese queries and returns answers from the handbook.  
- Supports local LLM inference (no external API key required).  
- Offers a web UI via Streamlit for easy interaction.

---

## ✨ Key Features

- 📄 **DOCX / PDF → Markdown**: Automatically cleans and normalizes handbook documents for indexing.  
- 🔎 **Hybrid Search**: Combines vector-based embeddings and BM25 for better recall & relevance.  
- 🇻🇳 **Vietnamese Language Support**: Uses Vietnamese embeddings to handle queries and documents in Vietnamese.  
- 🧠 **Local LLM Support**: Works offline / self-hosted — no need for external APIs.  
- 🖥️ **Simple UI**: Streamlit-based interface — easy to deploy and use.  
- ⚙️ **Flexible & Extensible**: Can be adapted to other RAG-based knowledge systems beyond just the TLU handbook.

---

## 📦 Tech Stack

- Python >= 3.9  
- Milvus (vector store) + BM25 for retrieval  
- Vietnamese embedding model (see `requirements.txt`)  
- Streamlit for UI  
- Other dependencies listed in `requirements.txt`

---

## 🧰 Getting Started

### Prerequisites

- Python 3.9+ (or compatible)  
- `pip` installed  
- Docker & Docker Compose (optional, if using containerized setup)  

## Setup & Run Locally


### 1. Clone the repo
```sh
git clone https://github.com/pminhtu24/TLU-Handbook-AI-Assistant.git
cd TLU-Handbook-AI-Assistant
```

### 2. Create virtual env
```sh
python -m venv venv
source venv/bin/activate
```

OR using conda
```sh
conda create -n rag_handbook
conda activate rag_handbook
conda install pip          
```

### 3. Install dependencies
```sh
pip install -r requirements.txt
```

### 4. Run Milvus (Docker recommended)
```sh
docker run -d --name milvus \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest
```

### 5. Pull a Vietnamese-friendly model (using Ollama)
```sh
ollama pull qwen3:4b-instruct
```
### 6. Start the app
```sh
streamlit run app.py  
```
