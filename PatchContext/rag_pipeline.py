import os
import time
import logging
import traceback
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from data_fetcher import GitHubDataFetcher

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load env variables
load_dotenv()

class RAGPipeline:
    """RAG pipeline using HuggingFace embeddings, FAISS vector store, and ChatGroq."""
    
    INDEX_PATH = "faiss_index"
    
    def __init__(self, owner: str = "fastapi", repo: str = "fastapi"):
        self.owner = owner
        self.repo = repo
        
        # Initialize embeddings model
        logger.info("Initializing HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        
        # Verify Groq API Key is present
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables. RAG pipeline query execution will fail.")

    def get_or_create_vectorstore(self, force_refresh: bool = False) -> FAISS:
        """Loads local FAISS vector store if exists, otherwise fetches data and creates one."""
        index_exists = os.path.exists(self.INDEX_PATH)
        
        if index_exists and not force_refresh:
            logger.info(f"Loading existing FAISS index from '{self.INDEX_PATH}'...")
            try:
                self.vectorstore = FAISS.load_local(
                    self.INDEX_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS index loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load local FAISS index: {e}. Rebuilding...")
                self._build_and_save_vectorstore()
        else:
            logger.info("FAISS index does not exist or force_refresh is True. Building index...")
            self._build_and_save_vectorstore()
            
        return self.vectorstore

    def _build_and_save_vectorstore(self):
        """Fetches data from GitHub, chunks them, builds and saves FAISS vector store."""
        fetcher = GitHubDataFetcher(owner=self.owner, repo=self.repo)
        documents = fetcher.fetch_all()
        
        if not documents:
            logger.warning("No documents fetched from GitHub. Creating empty/mock FAISS database to prevent crash.")
            # Create a fallback document so FAISS can initialize without crashing
            fallback_doc = Document(
                page_content="FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.",
                metadata={
                    "type": "issue", 
                    "title": "Fallback Document", 
                    "number": 0, 
                    "sha": None, 
                    "url": "", 
                    "author": "system", 
                    "created_at": ""
                }
            )
            documents = [fallback_doc]
            
        # Split documents
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} source documents into {len(chunks)} chunks.")
        
        # Build FAISS vector store
        logger.info("Generating embeddings and building FAISS vector store...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Save FAISS locally
        logger.info(f"Saving FAISS index to '{self.INDEX_PATH}'...")
        os.makedirs(self.INDEX_PATH, exist_ok=True)
        self.vectorstore.save_local(self.INDEX_PATH)
        logger.info("FAISS index saved successfully.")

    def setup_qa_chain(self):
        """Sets up the retriever, LLM, and RetrievalQA chain."""
        if not self.vectorstore:
            self.get_or_create_vectorstore()
            
        logger.info("Setting up MMR Retriever (k=5, fetch_k=20, lambda_mult=0.7)...")
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )
        
        # Verify Groq API Key is present
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY is not set. ChatGroq cannot be initialized. RAG queries will return error.")
            self.llm = None
            self.qa_chain = None
            return
            
        logger.info("Initializing ChatGroq LLM (llama-3.1-8b-instant)...")
        try:
            # Initialize Groq LLM (Strict: llama-3.1-8b-instant, temp=0.2)
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.2,
                groq_api_key=self.groq_api_key
            )
            
            # Define Prompt Template
            system_prompt = """You are PatchContext.
Only answer using the retrieved documents. Do not make up facts or use external knowledge.
If partial evidence exists in the retrieved documents, synthesize an answer using only the provided context.
Your primary goal is to explain WHY FastAPI was designed this way, focusing on the technical design decisions, rationales, trade-offs, and creator discussions.
Prefer design discussions, issues, and pull requests over generic definitions.
Always cite the Issue #, PR #, or Commit SHA of the source documents in your answer as evidence.
If you are uncertain or the context is incomplete, mention the uncertainty or missing details, but still summarize the available context.
Never fabricate or assume details that are not grounded in the retrieved documents.
If the retrieved documents do not contain any information relevant to the question, reply exactly:
"I couldn't find this in repository history."

Context:
{context}

Question: {question}
Answer:"""
            
            prompt = PromptTemplate(
                template=system_prompt,
                input_variables=["context", "question"]
            )
            
            logger.info("Initializing RetrievalQA chain...")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            logger.info("RAG pipeline setup completed.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq or RetrievalQA: {e}")
            self.llm = None
            self.qa_chain = None

    def query(self, question: str) -> Dict[str, Any]:
        """Runs query through RetrievalQA chain and returns answer, sources, and execution time."""
        if not self.qa_chain:
            self.setup_qa_chain()
            
        start_time = time.time()
        
        if not self.qa_chain:
            logger.info("ChatGroq QA chain is not active. Performing retrieval-only fallback.")
            source_docs = []
            if self.retriever:
                try:
                    source_docs = self.retriever.invoke(question)
                except Exception as e:
                    logger.error(f"Failed to retrieve documents during fallback: {e}")
            execution_time = time.time() - start_time
            return {
                "answer": "Error executing query. Please verify your GROQ_API_KEY in the .env file.",
                "source_documents": source_docs,
                "execution_time": execution_time
            }
            
        try:
            logger.info(f"Running query: '{question}'")
            
            retrieved_docs = []
            if self.retriever:
                try:
                    retrieved_docs = self.retriever.invoke(question)
                except Exception as ret_err:
                    logger.error(f"Retriever invoke failed: {ret_err}")
            
            # Get similarity scores using similarity_search_with_score
            scores = []
            max_similarity = 0.0
            scored_results = []
            if self.vectorstore:
                try:
                    # Search top 20 to find scores for our retrieved documents
                    scored_results = self.vectorstore.similarity_search_with_score(question, k=20)
                except Exception as score_err:
                    logger.error(f"Similarity search with score failed: {score_err}")
            
            for doc in retrieved_docs:
                best_l2 = None
                for s_doc, s_val in scored_results:
                    if s_doc.page_content == doc.page_content:
                        best_l2 = s_val
                        break
                if best_l2 is None:
                    # Fallback lookup directly
                    try:
                        single_res = self.vectorstore.similarity_search_with_score(doc.page_content, k=1)
                        if single_res:
                            best_l2 = single_res[0][1]
                    except Exception:
                        pass
                
                if best_l2 is not None:
                    # Cosine similarity mapping from L2 distance
                    similarity = 1.0 - (best_l2 ** 2) / 2.0
                    similarity = max(0.0, min(1.0, similarity))
                else:
                    similarity = 0.0
                
                scores.append(similarity)
                if similarity > max_similarity:
                    max_similarity = similarity
            
            # Check programmatic fallback condition
            fallback_triggered = False
            fallback_reason = ""
            if len(retrieved_docs) == 0:
                fallback_triggered = True
                fallback_reason = "No documents retrieved from vectorstore."
            elif max_similarity < 0.3:
                fallback_triggered = True
                fallback_reason = f"Max similarity score too low ({max_similarity:.4f} < 0.3 similarity_threshold)."
            
            # Print diagnostics as required by #5
            print(f"\n--- [DIAGNOSTICS] QUERY: '{question}' ---")
            print(f"Number of retrieved chunks: {len(retrieved_docs)}")
            print("Retrieved chunks similarity scores (Relevance):")
            for idx, (doc, sim) in enumerate(zip(retrieved_docs, scores), 1):
                title_safe = doc.metadata.get('title', 'N/A').encode('ascii', errors='replace').decode('ascii')
                print(f"  [{idx}] Similarity Score: {sim:.4f} | Type: {doc.metadata.get('type')} | Title: {title_safe}")
            print(f"Fallback triggered: {fallback_triggered}")
            if fallback_triggered:
                print(f"Reason for fallback: {fallback_reason}")
            print("-----------------------------------------\n")
            
            if fallback_triggered:
                execution_time = time.time() - start_time
                return {
                    "answer": "I couldn't find this in repository history.",
                    "source_documents": retrieved_docs,
                    "execution_time": execution_time
                }
            
            # --- Print Complete LLM Prompt ---
            context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
            system_prompt = """You are PatchContext.
Only answer using the retrieved documents. Do not make up facts or use external knowledge.
If partial evidence exists in the retrieved documents, synthesize an answer using only the provided context.
Your primary goal is to explain WHY FastAPI was designed this way, focusing on the technical design decisions, rationales, trade-offs, and creator discussions.
Prefer design discussions, issues, and pull requests over generic definitions.
Always cite the Issue #, PR #, or Commit SHA of the source documents in your answer as evidence.
If you are uncertain or the context is incomplete, mention the uncertainty or missing details, but still summarize the available context.
Never fabricate or assume details that are not grounded in the retrieved documents.
If the retrieved documents do not contain any information relevant to the question, reply exactly:
"I couldn't find this in repository history."

Context:
{context}

Question: {question}
Answer:"""
            complete_prompt = system_prompt.format(context=context_str, question=question)
            safe_prompt = complete_prompt.encode('ascii', errors='replace').decode('ascii')
            print("\n=== COMPLETE PROMPT SENT TO LLM ===")
            print(safe_prompt)
            print("===================================\n")
            
            # Execute QA chain
            result = self.qa_chain.invoke({"query": question})
            execution_time = time.time() - start_time
            
            return {
                "answer": result.get("result", "").strip(),
                "source_documents": result.get("source_documents", []),
                "execution_time": execution_time
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing query: {e}")
            return {
                "answer": "Error executing query. Please verify your GROQ_API_KEY in the .env file.",
                "source_documents": [],
                "execution_time": execution_time
            }

if __name__ == "__main__":
    # Test RAG Pipeline locally if run directly
    pipeline = RAGPipeline()
    pipeline.get_or_create_vectorstore()
    pipeline.setup_qa_chain()
    
    test_q = "What is FastAPI?"
    res = pipeline.query(test_q)
    print(f"Q: {test_q}")
    print(f"A: {res['answer']}")
    print(f"Time: {res['execution_time']:.4f}s")
    print(f"Sources count: {len(res['source_documents'])}")
