# Medical Information Chatbot

A question-answering chatbot built with LangChain and Hugging Face models that can answer medical queries based on The GALE Encyclopedia of Medicine.

## Features

- Uses FLAN-T5-small model for question answering
- FAISS vector store for efficient document retrieval
- Sentence transformers for document embeddings
- Custom prompt template for medical Q&A

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install langchain langchain-huggingface faiss-cpu python-dotenv
```

3. Place your medical document in the `data/` directory
4. Run the vectorstore creation:
```bash
python create_memory_for_llm.py
```

5. Start asking questions:
```bash
python connect_memory_with_llm.py
```

## Project Structure

- `connect_memory_with_llm.py`: Main QA interface
- `create_memory_for_llm.py`: Creates FAISS vectorstore from documents
- `data/`: Contains medical documents
- `vectorstore/`: Contains FAISS index files

## Models Used

- LLM: google/flan-t5-small
- Embeddings: sentence-transformers/all-MiniLM-L6-v2