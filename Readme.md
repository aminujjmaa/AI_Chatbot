# Document QA Bot 🤖

A powerful document analysis and question-answering system built with Streamlit and LangChain. This application allows users to upload various types of documents and interact with their content through natural language queries.

## Features ✨

- **Multi-Format Support** 📚

  - PDF documents
  - Excel spreadsheets (XLSX, XLS)
  - PowerPoint presentations (PPT, PPTX)
  - Images (PNG, JPG, JPEG)
  - Videos (MP4, WEBM, AVI, MOV)
  - SCORM packages
  - ZIP archives

- **Advanced Document Processing** 🔍

  - Table extraction and analysis
  - Image OCR (Optical Character Recognition)
  - Video transcription
  - Text chunking and semantic search
  - Financial data parsing

- **Interactive Chat Interface** 💬

  - Natural language queries
  - Context-aware responses
  - Session management
  - Chat history persistence

- **Data Visualization** 📊
  - Table display
  - Image preview
  - Structured data presentation
  - Processing status updates

## Installation 🚀

1. Clone the repository:

```bash
git clone https://github.com/yourusername/document-qa-bot.git
cd document-qa-bot
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file in the root directory with:

```env
groq_api=your_groq_api_key_here
```

## Usage 🎯

1. Start the application:

```bash
streamlit run src/app.py
```

2. Open your web browser and navigate to the provided URL (typically `http://localhost:8501`)

3. Upload your documents using the sidebar

4. Start asking questions about your documents!

## Features in Detail 🔎

### Document Processing

- **PDF Processing**: Extracts text and tables with multiple fallback methods
- **Excel Processing**: Handles complex spreadsheets and tabular data
- **Image Processing**: OCR for text extraction from images
- **Video Processing**: Transcription and audio analysis
- **SCORM Processing**: Extracts content from e-learning packages

### Question Answering

- Uses LangChain's RAG (Retrieval Augmented Generation) for accurate responses
- Maintains conversation context for follow-up questions
- Provides citations and references to source documents

### Data Handling

- Efficient chunking and vectorization of document content
- FAISS vector store for fast similarity search
- Session-based document and chat history management

## Technical Architecture 🏗️

- **Frontend**: Streamlit
- **Backend Processing**: LangChain, FAISS
- **Language Model**: Groq (Llama3-8b-8192)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Document Processing**:
  - PDFPlumber
  - PyPDFLoader
  - Tesseract OCR
  - Pandas
  - MoviePy

## Requirements 📋

- Python 3.8+
- Tesseract OCR (for image processing)
- FFmpeg (for video processing)
- Required Python packages listed in `requirements.txt`

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- LangChain for the RAG framework
- Streamlit for the web interface
- Groq for the language model API
- HuggingFace for embeddings
- All other open-source libraries used in this project

## Support 💪

If you encounter any issues or have questions, please file an issue on GitHub.
