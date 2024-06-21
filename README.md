# Gemma Model Document Q&A

This Streamlit application allows users to ask questions about documents using the Gemma language model and vector embeddings.

## Features

- Document ingestion from PDF files
- Text chunking and embedding
- Vector similarity search
- Question answering based on document content

## Requirements

- Python 3.7+
- Streamlit
- LangChain
- Groq (used LLaMa3-8b, can use others as well)
- Google Generative AI
- FAISS
- PyPDF2

## Installation

1. Clone this repository
2. Install The required Packages:
   pip install -r requirements.txt
3. Set up your environment variables:
   Create a `.env` file in the project root and add your API keys:
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key

## Usage

1. Place your PDF documents in the project directory(foder named PDF).

2. Run the Streamlit app:

streamlit run app.py

3. In the web interface:

- Click "Documents Embedding" to process and embed the documents.
- Enter your question in the text input field.
- View the answer and relevant document chunks.

## How it Works

1. The application loads PDF documents from a specified directory.
2. It splits the documents into smaller chunks.
3. These chunks are embedded using Google's Generative AI embeddings.
4. The embeddings are stored in a FAISS vector database for efficient similarity search.
5. When a question is asked, the most relevant document chunks are retrieved.
6. The Gemma model (via Groq) generates an answer based on the retrieved context.

## Customization

- Adjust the `chunk_size` and `chunk_overlap` in the `RecursiveCharacterTextSplitter` to change how documents are split.
- Modify the prompt template in the `ChatPromptTemplate` to alter how the model interprets questions and context.

## Troubleshooting

If you encounter issues:

- Ensure all API keys are correctly set in the `.env` file.
- Check that the PDF documents are in the correct directory and are readable.
- Verify that all dependencies are installed correctly.
