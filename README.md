# Automated_Information_Retrieval_System_using_LLM

**Overview**

This project is an AI-powered assistant built for equity research analysts. It automates the retrieval of relevant financial information from trusted sources and provides accurate, context-aware answers to user queries using a Retrieval-Augmented Generation (RAG) approach.

The system helps reduce manual effort in searching through large financial documents and delivers precise insights by leveraging advanced language models and vector search.

**Live App**

Access the deployed application here:

https://irsystem.streamlit.app/

**Key Features**

1. Automated Web Scraping of financial content

2. Contextual Retrieval using OpenAI Embeddings + FAISS

3. Fact-grounded Answer Generation via LLM (OpenAI/Gemini)

4. Streamlit Interface for seamless interaction

5. RAG Pipeline to minimize hallucination and maximize factuality

**Tech Stack**
- Frontend: Streamlit

- Backend: Python

- LLM APIs: OpenAI, Google Gemini

- Vector DB: FAISS

- Embeddings: OpenAI text-embedding-ada-002

- Web Scraping: beautifulsoup (preferred for dynamic websites)

- Framework: LangChain for document loading and RAG flow

How It Works
- Scrapes or loads financial content (reports, articles).

- Splits text into chunks and generates embeddings.

- Stores embeddings in a FAISS vector index.

- On query, retrieves top-k relevant chunks.

- Sends the query + context to an LLM to generate the final answer.

**Use Cases**
- Quickly extract insights from large equity reports

- Find company-specific financial data or trends

- Assist in market commentary preparation

- Reduce research time and improve decision-making
