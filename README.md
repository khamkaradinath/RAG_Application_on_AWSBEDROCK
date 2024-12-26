# Chat with PDF using AWS Bedrock

This project allows users to interact with PDF documents using AWS Bedrock. The application processes PDF files, creates vector embeddings, and uses language models to answer user queries based on the content of the PDFs.

## Workflow

1. **Data Ingestion**: Load PDF documents from the `data` directory and split them into chunks.
2. **Vector Embedding and Vector Store**: Create vector embeddings for the document chunks and store them in a FAISS index.
3. **Query Processing**: Use language models to process user queries and provide answers based on the vector embeddings.

### Detailed Steps

1. **Data Ingestion**
    - Load PDF documents using `PyPDFDirectoryLoader`.
    - Split documents into chunks using `RecursiveCharacterTextSplitter`.

2. **Vector Embedding and Vector Store**
    - Create vector embeddings using `BedrockEmbeddings`.
    - Store embeddings in a FAISS index.

3. **Query Processing**
    - Load the FAISS index.
    - Use language models (`Claude` and `Llama2`) to process user queries.
    - Retrieve and display answers based on the vector embeddings.

### Flowchart

```mermaid
graph TD;
    A[Start] --> B[Load PDF Documents]
    B --> C[Split Documents into Chunks]
    C --> D[Create Vector Embeddings]
    D --> E[Store Embeddings in FAISS Index]
    E --> F[User Query]
    F --> G[Load FAISS Index]
    G --> H[Process Query with Language Model]
    H --> I[Retrieve and Display Answer]
    I --> J[End]