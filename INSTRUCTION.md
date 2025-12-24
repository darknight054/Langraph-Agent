AI Engineer Technical Assignment 
Objective 
Design and implement a Retrieval-Augmented Generation (RAG) system that ingests documents, extracts text using OCR,  
and enables users to chat with the document content. The system must be orchestrated using an agentic workflow built  
with LangGraph. 
This assignment evaluates your ability to design real-world AI systems involving: - Unstructured document ingestion 
- OCR-based text extraction 
- Vector search 
- LLM-based question answering 
- Agentic control flow using LangGraph 
Assignment Overview 
High-Level Requirements 
1. Document Ingestion 
The system should support ingestion of: 
- Standard PDF documents 
- Scanned PDFs or image-based documents 
OCR Technology (Mandatory): 
- Use DeepSeek OCR for document text extraction.
- If DeepSeek OCR is unavailable, a mocked interface is acceptable, but the design and code must clearly reflect integration points. 
2. Text Processing 
- Clean and normalize extracted text. 
- Chunk documents into semantically meaningful segments for embedding. 
3. Embeddings & Vector Store 
Use one of the following vector databases: 
- ChromaDB 
- FAISS 
The vector store must persist embeddings and retrieve relevant document chunks during question-answering. 
4. RAG Pipeline with LangGraph (Mandatory) 
The chat system must be orchestrated using LangGraph to create an agentic workflow. 
Required Agents / Nodes: 
- Retriever Agent: Fetches relevant document chunks from the vector store. - Generator Agent: Uses an LLM to generate answers based on retrieved context. - Validator Agent: Evaluates the generated answer for relevance and hallucinations. - Final Response Agent: Returns the validated answer to the user. 
Workflow Logic: 
- Shared state management.
- Conditional transitions (logic to decide the next node). 
- At least one retry loop if validation fails (e.g., hallucinatory answers). 5. Language Model 
Use one of the following LLMs: 
- Google Gemini 
- OpenAI GPT models 
- Open-source LLMs (local via Ollama or hosted) 
Configuration: 
- Use environment variables (e.g., .env file) for API keys. 
- Strictly avoid hardcoded credentials. 
Deliverables 
1. Source Code Repository: Complete codebase with clear structure. 2. README File: 
 - System architecture overview. 
 - Technologies and libraries used. 
 - Step-by-step setup and execution instructions. 
3. Sample Data: 
 - A sample document. 
 - A transcript of an example chat interaction demonstrating the system. Optional Bonus 
- Build a simple Streamlit app to interact with the RAG system. - Include features like document upload, chat interface, and response display.
Evaluation Criteria 
Criteria Weightage 
Code Quality & Structure 25% 
Functionality & Accuracy 30% 
Agentic Workflow Design 20% 
Documentation & Readability 15% 
Optional Streamlit App 10% 
Submission Guidelines 
- Submit a GitHub/GitLab repository link with the code and README. - Ensure the repository is public or accessible to the evaluator. - Include a short video demo (optional but recommended) showcasing the system. Timeline 
Deadline: 48 hours from receipt of the assignment.
