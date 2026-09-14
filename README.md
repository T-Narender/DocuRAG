# DocuRAG

DocuRAG is a Streamlit document question-answering application. Upload a document, let the app extract and index its text, and ask questions about the uploaded content using a retrieval-augmented generation (RAG) pipeline.

## Features

- Upload PDF, TXT, DOCX, DOC, PPTX, and PPT files.
- Extract text with LangChain document loaders.
- Split documents into 500-character chunks with 100-character overlap.
- Create local embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Store the active document's vectors in a local Chroma database.
- Retrieve the three most relevant chunks for each question.
- Generate answers with Google's Gemini 2.5 Flash model.
- Display source file and page references with each answer.
- Keep recent conversation history while the uploaded document remains active.

## How It Works

1. A file is uploaded through the Streamlit sidebar.
2. The file is written to a temporary location and loaded with the appropriate parser.
3. The extracted text is split into overlapping chunks and embedded locally.
4. The Chroma directory is recreated for the new upload.
5. Questions retrieve the most relevant chunks and send them, together with recent chat history, to Gemini.

Only one document is active at a time. Uploading a different file clears the current conversation and rebuilds the local index.

## Requirements

- Python 3.11 or newer
- A Google Gemini API key
- Internet access for Gemini requests and the first download of the embedding model
- Optional: Docker Desktop for containerized use

## Local Setup

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
```

The `.env` file is ignored by Git. Never commit a real API key or place one in the README.

Start the application:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`. You may also enter a Gemini API key directly in the sidebar for a single run.

## Docker

Build the image expected by `compose.yaml`:

```bash
docker build -t docurag:cpu .
```

Create `.env` as shown above, then start the service:

```bash
docker compose up
```

The application is available at `http://localhost:8501`. Stop it with `Ctrl+C`, or run it in the background with `docker compose up -d` and stop it with `docker compose down`.

The Docker image installs the CPU-only PyTorch build. Embedding inference therefore runs locally on the CPU, while answer generation uses the Gemini API.

## Using the App

1. Open the application.
2. Upload a supported document from the sidebar.
3. Wait for extraction, embedding, and indexing to finish.
4. Ask a question in the chat box.
5. Review the answer and its displayed source reference.

For reliable answers, ask questions that can be answered from the uploaded document. Scanned PDFs without an extractable text layer may need OCR before upload.

## Project Structure

| Path               | Purpose                                                                                |
| ------------------ | -------------------------------------------------------------------------------------- |
| `app.py`           | Streamlit interface and RAG pipeline                                                   |
| `requirements.txt` | Python dependencies                                                                    |
| `Dockerfile`       | CPU-based container image definition                                                   |
| `compose.yaml`     | Docker Compose service configuration                                                   |
| `rag1.ipynb`       | Earlier notebook experiments for document loading, chunking, embeddings, and retrieval |
| `.env`             | Local Gemini configuration; ignored by Git                                             |

Generated Chroma databases, Python environments, caches, and serialized experiment data are ignored by `.gitignore` and are not required to start the app. The active upload index is rebuilt when a document is uploaded.

## Troubleshooting

- **Missing API key:** Set `GEMINI_API_KEY` in `.env`, restart Streamlit, or enter a key in the sidebar.
- **Gemini rate limit:** Wait for the quota window to reset or use a key with available quota. The app retries temporary rate-limit errors automatically.
- **Upload processing fails:** Confirm that the file is one of the supported formats and contains extractable text.
- **Docker Compose cannot find `docurag:cpu`:** Build the image first with `docker build -t docurag:cpu .`.
- **Slow first request:** The embedding model is downloaded the first time it is initialized.

## Security Notes

- Keep `.env` private and verify it is ignored before pushing to GitHub.
- API requests to Gemini contain the retrieved document context and question. Do not upload confidential documents unless that use complies with your organization's data policy.
- Uploaded files are temporarily written for parsing, and the active vector index is stored locally in `chroma_db_upload/`.

## License

No license has been specified for this repository yet. Add a license before distributing the project if one is required for your use case.
