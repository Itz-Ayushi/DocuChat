# 📄 DocuChat: Multi-Doc AI Assistant

**DocuChat** is a powerful, AI-driven research assistant that allows users to upload multiple PDF and DOCX documents and explore them through natural language queries. Built with Streamlit and Google Gemini, it uses Retrieval-Augmented Generation (RAG) to deliver accurate summaries and context-aware responses based solely on your uploaded documents.

---

## 🚀 Features

- **Multi-Document Support:** Upload and analyze multiple PDF and DOCX files at once.
- **Instant Summarization:** Automatically generates a concise summary of uploaded content.
- **Context-Aware Chat:** Get precise answers pulled directly from your documents.
- **Privacy-Focused:** All processing happens in a temporary session—no permanent storage.
- **Fast & Efficient:** Uses PyMuPDF for fast PDF parsing and ChromaDB for vector search.

---

## 🛠 Tech Stack

**Frontend:** Streamlit  
**LLM:** Google Gemini 1.5 Flash (`google-generativeai`)  
**Vector Database:** ChromaDB  
**Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)  
**PDF Parsing:** PyMuPDF (`fitz`)  
**DOCX Parsing:** python-docx  

---

# ⚙ Installation & Setup

Follow these steps to run the application locally.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Itz-Ayushi/DocuChat.git
```

## 2. Create a Virtual Environment (Recommended)

It’s best practice to use a virtual environment to manage dependencies.

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 4. Configure API Key

You will need a Google Gemini API key to run the app.

Steps:

Go to Google AI Studio and generate an API key.

In the project root, create a folder named:
```bash
.streamlit
```


Inside that folder, create a file named:
```bash
secrets.toml
```


Add your API key:
```bash
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_actual_api_key_here"
```

# ▶️ Usage

Start the application:
```bash
streamlit run app.py
```

Once running:

The app will open in your default browser (http://localhost:8501
).

Use the Upload Documents sidebar option to upload PDF/DOCX files.

Click Process & Summarize.

Wait for the summary to appear.

Ask questions in the chat box to query content from your documents.

---
# 🧠 How It Works

Ingestion: Extracts text from uploaded files.

Chunking: Splits text into overlapping chunks to retain context.

Embedding: Converts chunks into vector embeddings using all-MiniLM-L6-v2.

Storage: Temporarily stores embeddings in an in-memory ChromaDB collection.

Retrieval: Retrieves the most relevant chunks when you ask a question.

Generation: Sends chunks + your query to Gemini, which generates an answer strictly from document context.

# 🤝 Contributing

Contributions are welcome!
Feel free to submit issues or pull requests to improve DocuChat.
```bash
git clone https://github.com/Ishita-01/DocuChat.git
