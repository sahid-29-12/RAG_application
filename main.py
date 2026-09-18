import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

st.set_page_config(page_title="RAG Document Q&A System", layout="wide")
st.title("RAG-Based Document Q&A System 📄🔍")
st.markdown("Upload a document, and ask questions based strictly on its content. This demonstrates the full RAG pipeline.")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
st.sidebar.markdown("[Get a Gemini API Key here](https://aistudio.google.com/app/apikey)")

uploaded_file = st.sidebar.file_uploader("Upload a document (PDF, TXT)", type=["pdf", "txt"])

def get_text_from_file(file):
    text = ""
    if file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    elif file.name.endswith(".txt"):
        text = file.getvalue().decode("utf-8")
    return text

if uploaded_file and api_key:
    with st.spinner("Ingesting and processing document..."):
        raw_text = get_text_from_file(uploaded_file)
        
        st.header("Step 1: Document Processing")
        with st.expander("View Extracted Text"):
            st.text(raw_text[:2000] + "\n\n... (Truncated for display)" if len(raw_text) > 2000 else raw_text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(raw_text)
        
        st.header("Step 2: Text Chunking")
        with st.expander(f"View Chunks (Total Chunks: {len(chunks)})"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}:**\n{chunk}\n---")

        st.header("Step 3 & 4: Embeddings and Vector Database")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        docs = [Document(page_content=t) for t in chunks]
        vector_store = FAISS.from_documents(docs, embeddings)
        st.success("Successfully generated embeddings and stored them in the FAISS Vector Database!")

    st.header("Step 5 & 6: Similarity Search & LLM Generation")
    user_query = st.text_input("Ask a question about the document:")
    
    if user_query:
        docs_and_scores = vector_store.similarity_search_with_score(user_query, k=3)
        
        with st.expander("View Retrieved Context (Similarity Search Results)"):
            for i, (doc, score) in enumerate(docs_and_scores):
                st.markdown(f"**Rank {i+1}** (L2 Distance: `{score:.4f}`):\n> {doc.page_content}")
                
        prompt_template = """
        You are a helpful assistant answering questions based strictly on the provided context.
        Answer the question using the provided context. If the answer cannot be found in the 
        context, clearly state that "The information is not available in the document."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        retrieved_docs = [doc for doc, score in docs_and_scores]
        
        with st.spinner("Generating answer..."):
            response = chain.invoke({"input_documents": retrieved_docs, "question": user_query})
            st.markdown("### Final Answer:")
            st.info(response["output_text"])

elif not api_key:
    st.warning("Please enter your Google Gemini API Key in the sidebar.")
elif not uploaded_file:
    st.info("Please upload a document (PDF or TXT) in the sidebar to get started.")
