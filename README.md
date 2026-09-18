# RAG-Based Document Q&A System

## 1. Project Overview
This project is a simple **Retrieval-Augmented Generation (RAG)** application built with Streamlit. It allows users to upload a document (PDF or TXT) and ask questions about its content. The system extracts the text, breaks it into smaller chunks, converts them into vector embeddings, stores them in a local vector database, and uses similarity search to retrieve the most relevant context. An LLM (Google Gemini) then formulates a factual answer strictly based on the retrieved context.

## 2. Architecture / Workflow
1. **Document Ingestion**: User uploads a document via the Streamlit UI.
2. **Text Extraction**: PyPDF2 or UTF-8 decoding extracts raw text.
3. **Chunking**: `RecursiveCharacterTextSplitter` breaks the text into manageable chunks.
4. **Embeddings**: `HuggingFaceEmbeddings` converts text chunks into mathematical vectors.
5. **Vector Database**: `FAISS` stores the vectors for high-speed indexing.
6. **Retrieval**: User queries are vectorized and compared against the DB using L2 distance similarity search to find the top matching chunks.
7. **Generation**: The top chunks are injected into a rigid prompt and sent to Google Gemini (`gemini-1.5-flash`), which returns the final answer.

## 3. Technologies Used
* **Frontend UI**: Streamlit
* **Orchestration**: LangChain
* **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`) via `sentence-transformers`
* **Vector Database**: FAISS (Facebook AI Similarity Search)
* **LLM**: Google Gemini API (`langchain-google-genai`)
* **Document Parsing**: `PyPDF2`

## 4. Setup Instructions
1. Clone or download this repository.
2. Ensure you have Python 3.8+ installed on your system.
3. Open a terminal/command prompt in the project folder.
4. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 5. How to run the application
1. Run the Streamlit app from your terminal:
   ```bash
   streamlit run app.py
   ```
2. Open the URL provided in the terminal (usually `http://localhost:8501`).
3. Enter your Google Gemini API Key in the sidebar.
4. Upload the provided `sample_policy.txt` or any PDF document.
5. Follow the UI to see the extraction, chunking, and querying process in action.

## 6. Sample document used for testing
A sample document named `sample_policy.txt` is included in this repository. It contains a mock "Company Work-From-Home (WFH) Policy" detailing eligibility, equipment stipends, working hours, and security protocols.

---

## Explanations (Assignment Deliverables 16-21)

### 7. & 8. What embeddings are
Embeddings are numerical representations (lists of floating-point numbers/vectors) of text. They capture the semantic meaning and context of words, sentences, or paragraphs in a high-dimensional mathematical space. Words or sentences with similar meanings are placed closer together in this space.

### 9. Why a vector database is required
Traditional relational databases search for exact keyword matches. Vector databases (like FAISS or ChromaDB) are specifically designed to store high-dimensional embeddings and perform mathematical distance calculations rapidly. This allows the system to search by "concept" or "meaning" rather than relying on exact word matches.

### 10. How similarity search works
When a user asks a question, the query is passed through the same embedding model to create a "query vector." The vector database then calculates the distance (using metrics like Cosine Similarity or L2 Euclidean distance) between the query vector and all the stored document chunk vectors. The chunks with the shortest distance (highest similarity) are returned as the most relevant context.

### 11. What chunk size and overlap you selected
* **Chunk Size: 500 characters** 
* **Chunk Overlap: 100 characters**
* **Why**: A chunk size of 500 is large enough to contain complete thoughts or rules (like a single policy paragraph) without diluting the semantic meaning of the vector. The 100-character overlap ensures that sentences crossing the chunk boundary aren't abruptly cut off, preserving context for the LLM.

### 12. How RAG differs from simply asking an LLM a question
When you simply ask an LLM a question, it relies entirely on its pre-trained internal knowledge, which may be outdated, generalized, or prone to "hallucinations" (making things up). **RAG (Retrieval-Augmented Generation)** forces the LLM to read specific, highly relevant, and private/up-to-date information (the retrieved chunks) and bases its answer *only* on that injected context. This makes the output factual, traceable, and customized to your specific documents.

---

### Bonus Challenges Implemented
* Added a Streamlit web-based UI.
* Allowed users to upload PDFs and TXT dynamically.
* Displayed source chunks and relevance/similarity (L2 distance) scores.
* Implemented the "I don't know" response guardrail in the prompt template.
