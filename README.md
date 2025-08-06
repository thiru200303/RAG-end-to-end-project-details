# Local RAG Chat Assistant with Multimodal AI & Offline Function Calling API

### Overview

- A fully offline intelligent assistant that can:

- Read and answer questions from PDFs

- Understand images and audio

- Extract text from scanned files

- Perform basic offline function calling (like telling time or weather)

- All powered by local models with zero internet dependency

> Inspired by tools like ChatPDF, ChatGPT, and NotebookLM, but built for privacy and flexibility.

### How It Works?

**This assistant uses RAG (Retrieval-Augmented Generation) architecture combined with multimodal AI:**

1. PDF / Audio / Image is Uploaded
  - Your data is stored in the /data folder.

2. Text is Extracted

- PDF → Text chunks via PyMuPDF

- Audio → Transcribed with Whisper

- Image → Captioned by BLIP or Text extracted via OCR

3. Embedding is Generated
  - Text chunks are converted to embeddings using MiniLM.

4. Data is Stored in FAISS
  - The vector store is searchable using semantic similarity.

5. You Ask a Question
  - Your question is also embedded → matched with top document chunks from FAISS.

6. Answer is Generated
  - Relevant chunks + your question are sent to Mistral (via Ollama) for answer generation.

7. Function Calling (Offline)
  - When your query matches a certain pattern (like “time”, “weather”), it runs a Python function locally to return the result.
  
  
### Features (with Explanations):

| Feature                                             | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🧠 **RAG with Local LLM (Mistral via Ollama)**      | This project uses **Retrieval-Augmented Generation (RAG)** to answer your questions. When you ask a question about a document, it doesn't try to "memorize" the answer. Instead, it:<br> 1. Splits your PDFs into chunks <br> 2. Converts them to embeddings (numerical form of meaning) <br> 3. Searches for the most relevant chunks using FAISS <br> 4. Feeds those chunks + your question into Mistral (a local LLM) <br> ✅ This is how tools like ChatPDF find document-specific answers. |
| 🔎 **Embeddings with MiniLM**                       | It uses a small, fast model called `all-MiniLM` to convert your text (PDF chunks or question) into **vectors**. These vectors are what allow the assistant to **"understand meaning"** and search through your documents efficiently.                                                                                                                                                                                                                                                           |
| ⚡ **FAISS for Fast Retrieval**                      | After generating vector embeddings, it stores them in a **FAISS vector database**. When you ask a question, FAISS is used to find **the most relevant content from the documents** – super fast, even offline.                                                                                                                                                                                                                                                                                  |
| 📄 **PDF Document Querying**                        | You can upload PDFs into the `/data/` folder. The assistant reads them using **PyMuPDF**, splits them into smaller chunks, embeds them, and makes them searchable. When you ask questions, it answers by pulling from the content of the PDF.                                                                                                                                                                                                                                                   |
| 🎧 **Offline Audio Transcription with Whisper**     | If you drop an audio file (e.g., `.mp3` or `.wav`) into `/data/`, the assistant uses **Whisper (a speech-to-text model)** to transcribe it into text. Then, it can answer questions or summarize the audio file.                                                                                                                                                                                                                                                                                |
| 🖼️ **Offline Image Captioning with BLIP**          | You can upload an image (e.g., `.jpg`, `.png`) and ask “What’s in this image?” The assistant uses the **BLIP model** (Bootstrapped Language Image Pretraining) to generate a **natural language caption** for the image. This mimics how ChatGPT or ChatPDF with vision understands images.                                                                                                                                                                                                     |
| 🔠 **Text Extraction from Images (OCR)**            | If the image contains text (like a scanned book page), the assistant uses **Tesseract OCR** to read that text. It can then summarize, translate, or answer questions about it — similar to ChatGPT with image reading.                                                                                                                                                                                                                                                                          |
| 🧠 **Local Function Calling (Weather, Time, etc.)** | The assistant supports **offline "function calling"**. When you ask “What time is it?” or “What’s the weather in Delhi?”, it doesn't go online. It detects the intent and runs a **local Python function** that gives you a mock or local value. Just like OpenAI's function calling, but offline!                                                                                                                                                                                              |
| 🛡️ **Fully Offline & Private**                     | Unlike ChatPDF or ChatGPT, nothing leaves your computer. All models run locally:<br> - LLM (Mistral via Ollama)<br> - Whisper (audio)<br> - BLIP (image)<br> - Embeddings + FAISS (retrieval)<br> ✅ Safe for research, corporate, and academic use.                                                                                                                                                                                                                                             |

## 🗂️ Folder Structure:

```
/local-ai-assistant
│
├── /data/ # Store PDFs, images, audio
├── /models/ # (Optional) Pre-downloaded models
├── app.ipynb # Main notebook to run the assistant
├── rag_pipeline.py # Core logic for retrieval + LLM response
├── function_tools.py # Offline function-calling logic
├── requirements.txt # All dependencies
└── README.md # You are here

```


---

## ⚙️ Installation & Setup

### 1. 🔧 Install Python packages

```
pip install -r requirements.txt

```
### 2. Install Ollama and Mistral model
- Download from https://ollama.com

- Run in terminal:

> ollama run mistral

### 3. Install Tesseract OCR (for text from images)
- Download installer: https://github.com/tesseract-ocr/tesseract

- After installation, update this in your script:

> pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


### Libraries & Tools Used:

| Category               | Library / Tool                    | Purpose                               |
|------------------------|-----------------------------------|----------------------------------------|
| 🔍 Embeddings          | `sentence-transformers` (MiniLM)  | Convert text to vectors                |
| 🔎 Vector DB           | `faiss`                           | Fast retrieval of relevant chunks      |
| 🧠 LLM                 | `ollama` + Mistral                | Local language model for generation    |
| 📄 PDF Parsing         | `PyMuPDF (fitz)`                  | Extracts text from PDF files           |
| 🎧 Audio Transcription | `openai-whisper`                  | Converts speech to text offline        |
| 🖼️ Image Captioning   | `Salesforce BLIP`                 | Generates captions for images          |
| 🔤 OCR                 | `pytesseract`                     | Extracts text from images              |
| 📸 Image Processing    | `Pillow`, `opencv-python`         | Image handling                         |
| 📅 Date/Time           | `datetime`                        | Used in offline function calls         |
| 🌤️ Function Tools     | Custom Python                     | Offline APIs for weather, time, etc.   |
| 📦 UI (optional)       | `Streamlit` or `Jupyter Notebook` | User interaction or testing interface  |


# 📚 Dependency Breakdown for Offline AI Assistant

This project relies on a set of powerful Python libraries for local language modeling, retrieval, document parsing, image/audio processing, and building an interactive offline assistant.

---

## 🧠 LLM, Embeddings & Retrieval Libraries

### 1. `torch>=2.0.0`
- **Purpose**: Core deep learning backend used by many libraries.
- **Usage**:
  - Required by `sentence-transformers`, `Whisper`, `TTS`, `BLIP`.
  - Enables GPU acceleration (optional).

---

### 2. `transformers>=4.30.0`
- **Purpose**: Hugging Face Transformers for using LLMs like Mistral, Falcon, etc.
- **Usage**:
  - Load and run local models (if not using Ollama).
  - Function calling and LLM inference in custom workflows.

---

### 3. `sentence-transformers>=2.2.2`
- **Purpose**: Embeds text (chunks + queries) into dense vectors.
- **Usage**:
  - Generate embeddings for PDFs and questions.
  - Used with FAISS for semantic retrieval.

---

### 4. `faiss-cpu>=1.7.4`
- **Purpose**: Facebook AI Similarity Search.
- **Usage**:
  - Store and retrieve embedded vectors.
  - Retrieves top-k relevant chunks for your query.

---

## 📄 PDF & Document Parsing

### 5. `pymupdf>=1.23.0` (aka `fitz`)
- **Purpose**: Fast and lightweight PDF parser.
- **Usage**:
  - Extract text and image blocks from PDFs.
  - Detect content structure for downstream processing.

---

### 6. `pdfplumber>=0.10.2`
- **Purpose**: High-fidelity PDF parsing for tabular/column layouts.
- **Usage**:
  - Extract structured data like tables.
  - Complements `pymupdf` for advanced parsing.

---

### 7. `python-docx>=0.8.11`
- **Purpose**: DOCX (Word) document parser.
- **Usage**:
  - Enables parsing `.docx` alongside PDFs for multimodal support.

---

## 🖼️ Image & OCR Processing

### 8. `pytesseract>=0.3.10`
- **Purpose**: OCR using Tesseract engine.
- **Usage**:
  - Extracts text from scanned documents or images.
  - Supports document QA on scanned PDFs.

---

### 9. `Pillow>=9.5.0`
- **Purpose**: Image processing library.
- **Usage**:
  - Resize, crop, save images.
  - Preprocess inputs for OCR or captioning.

---

### 10. `opencv-python>=4.8.0`
- **Purpose**: Advanced computer vision toolkit.
- **Usage**:
  - Preprocess and clean images before OCR.
  - Segment and analyze visual documents.

---

## 🎤🔊 Audio Processing and Speech

### 11. `openai-whisper>=20231117`
- **Purpose**: Local speech-to-text transcription.
- **Usage**:
  - Transcribe `.mp3` or `.wav` audio to text.
  - Enables voice input and audio Q&A.

---

### 12. `TTS>=0.20.2`
- **Purpose**: Text-to-Speech system.
- **Usage**:
  - Convert assistant responses into audible speech.

---

## 📊 Data Science & Visualization

### 13. `matplotlib>=3.7.1`
- **Purpose**: Plotting library.
- **Usage**:
  - Visualize embedding distributions or document stats.
  - Debug retrieval results.

---

### 14. `scikit-learn>=1.3.0`
- **Purpose**: Machine learning utilities.
- **Usage**:
  - Cosine similarity calculations.
  - Evaluate or visualize embedding clusters.

---

### 15. `numpy>=1.25.0`
- **Purpose**: Numerical computing.
- **Usage**:
  - Backbone for vector math in embedding and retrieval.
  - Used across most libraries.

---
# In Future:

## 🌐 Web App Interface + API Helpers

### 16. `streamlit>=1.28.0`
- **Purpose**: ML web app builder.
- **Usage**:
  - Create user-friendly UIs for PDF/image/audio chat.
  - Drag-and-drop files, view responses interactively.

---

### 17. `requests>=2.31.0`
- **Purpose**: HTTP request library.
- **Usage**:
  - Interact with local APIs (e.g., Ollama).
  - Trigger custom function tools or agents.

---

### 18. `python-dotenv>=1.0.0`
- **Purpose**: Manage secrets and environment configs.
- **Usage**:
  - Load `.env` for model paths, debug flags, keys, etc.

---

### 19. `sentencepiece>=0.1.99`
- **Purpose**: Tokenizer used in multilingual models.
- **Usage**:
  - Required by certain Hugging Face models (e.g., T5, mBERT).
  - Ensures compatibility with encoder-decoder LLMs.

---

##  Summary: RAG Workflow by Task

| **Task**                  | **Key Libraries**                                                                 |
|---------------------------|------------------------------------------------------------------------------------|
| Document Reading          | `pymupdf`, `pdfplumber`, `python-docx`, `pytesseract`, `Pillow`                   |
| Embedding & Retrieval     | `sentence-transformers`, `faiss-cpu`, `numpy`, `scikit-learn`                     |
| LLM Inference             | `transformers`, `torch`, `sentencepiece`                                          |
| Audio Transcription       | `openai-whisper`, `torch`                                                         |
| Image Handling            | `opencv-python`, `Pillow`, `pytesseract`                                          |
| TTS Response              | `TTS`, `torch`                                                                     |
| Interface                 | `streamlit`, `requests`, `python-dotenv`                                          |
| Visualization & Debug     | `matplotlib`, `scikit-learn`, `numpy`                                             |

---

> 🛡️ All libraries are compatible with fully offline setups. GPU acceleration is optional but recommended for Whisper, TTS, and LLMs.

### Models Used:

| Task             | Model/Tool           |
| ---------------- | -------------------- |
| LLM              | Mistral (via Ollama) |
| Embedding        | `all-MiniLM`         |
| PDF Reading      | PyMuPDF              |
| Vector Store     | FAISS                |
| Audio Transcribe | Whisper (local)      |
| Image Captioning | BLIP                 |
| Image OCR        | pytesseract + PIL    |

### For Beginners
**Even if you're new:**

- Just install dependencies ✅

- Download and run Mistral in Ollama ✅

- Put PDFs/audio/images in /data/ ✅

- Use the notebook like a chatbot ✅

**This project is a great way to learn about:**

- Local LLMs

- RAG Architecture

- Embeddings & FAISS

- Multimodal AI

### 📬 Contact / Contribute
**Want to extend this? Add:**

- Offline calendar search

- File search in PC

- Summarization dashboard

**Built by: Thiruvarasu**


---

## ✅ `requirements.txt`

```txt
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
pymupdf>=1.23.0
pytesseract>=0.3.10
Pillow>=9.5.0
openai-whisper>=20231117
opencv-python>=4.8.0
matplotlib>=3.7.1
scikit-learn>=1.3.0
python-docx>=0.8.11
pdfplumber>=0.10.2
streamlit>=1.28.0
requests>=2.31.0
sentencepiece>=0.1.99
TTS>=0.20.2
numpy>=1.25.0
python-dotenv>=1.0.0
```