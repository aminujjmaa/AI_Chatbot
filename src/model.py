import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from handlers import handle_pdf, handle_excel, handle_powerpoint, handle_image, handle_scorm, extract_video_text
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import pandas as pd
import re
from datetime import datetime
from langchain.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
import tabula
from tabulate import tabulate
from pdfplumber import open as plumber_open
import pdfplumber
import numpy as np

# Load environment variables
load_dotenv()

# Validate API Key
groq_api = os.getenv("groq_api")
if not groq_api:
    raise ValueError("The 'groq_api' key is not set in the environment variables.")

# Initialize embeddings and language model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=groq_api, model_name="Llama3-8b-8192")

# Extract structured P&L data from PDFs
def parse_document_data(docs, tables_data):
    """Parse both tabular and text data from documents"""
    structured_data = []
    text_content = []
    
    # Process tables first
    for table in tables_data:
        if not table.get("raw_data"):
            continue
            
        raw_data = table["raw_data"]
        if not raw_data or len(raw_data) < 2:  # Need at least headers and one data row
            continue
            
        headers = raw_data[0]
        data_rows = raw_data[1:]
        
        # Process each data row
        for row_idx, row in enumerate(data_rows):
            if not row:
                continue
                
            # First column is usually the row header/description
            row_header = row[0] if row else ""
            
            # Process each cell in the row
            for col_idx, value in enumerate(row[1:], 1):
                if not value:
                    continue
                    
                try:
                    # Get column header
                    col_header = headers[col_idx] if col_idx < len(headers) else f"Column_{col_idx}"
                    
                    # Create key combining row and column headers
                    key = f"{row_header} ({col_header})" if row_header else col_header
                    
                    # Clean and process the value
                    value_str = str(value).strip()
                    cleaned_value = value_str.replace('₹', '').replace('$', '').replace(',', '').strip()
                    
                    try:
                        # Try to convert to numeric
                        numeric_value = pd.to_numeric(cleaned_value, errors='coerce')
                        if pd.notna(numeric_value):
                            data_point = {
                                "key": key,
                                "value": numeric_value,
                                "original_value": value_str,
                                "row_header": row_header,
                                "col_header": col_header,
                                "page": table["page"],
                                "table_id": table["table_id"],
                                "type": "numeric",
                                "source": table["source"]
                            }
                            structured_data.append(data_point)
                            continue
                    except:
                        pass
                    
                    # Store as text if not numeric
                    text_content.append({
                        "page": table["page"],
                        "content": f"{key}: {value_str}",
                        "source": table["source"],
                        "section": "table",
                        "table_id": table["table_id"]
                    })
                except Exception as e:
                    print(f"Error processing table cell: {str(e)}")
                    continue
    
    # Process regular text content
    for doc in docs:
        if doc.metadata.get("content_type") != "table":
            text = doc.page_content
            page_num = doc.metadata.get("page", 0)
            
            # Clean and process text
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            if text:
                text_content.append({
                    "page": page_num,
                    "content": text,
                    "source": doc.metadata.get("source", "Unknown"),
                    "section": identify_section(text)
                })
    
    # Create DataFrames
    numbers_df = pd.DataFrame(structured_data) if structured_data else pd.DataFrame(columns=[
        "key", "value", "original_value", "row_header", "col_header", 
        "page", "table_id", "type", "source"
    ])
    
    text_df = pd.DataFrame(text_content)
    
    return {
        "numerical_data": numbers_df,
        "text_content": text_df,
        "tables": tables_data
    }

def identify_section(text):
    """Helper function to identify document sections"""
    section_patterns = {
        "header": r'^(?:Title|Header|Introduction|Abstract)',
        "table": r'(?:Table|Figure)\s+\d+',
        "list": r'^\s*[•\-\d]+\.',
        "paragraph": r'^[A-Z][^.!?]{10,}[.!?]'
    }
    
    for section_type, pattern in section_patterns.items():
        if re.search(pattern, text, re.MULTILINE):
            return section_type
    return "general"

def process_match(match, pattern, page_num):
    """Helper function to process regex matches"""
    if not match or not match[0].strip():
        return None
        
    data_point = {
        "context": match[0].strip(),
        "page": page_num,
        "raw_text": ' '.join(str(m) for m in match).strip(),
        "type": "numeric"
    }
    
    # Process different types of matches
    if '%' in pattern:
        data_point.update({
            "percentage": float(match[1]) if match[1].strip() else None,
            "type": "percentage"
        })
    elif 'to|-' in pattern:
        data_point.update({
            "range_start": float(match[1].replace(',', '')),
            "range_end": float(match[2].replace(',', '')),
            "type": "range"
        })
    else:
        try:
            value = float(match[-1].replace(',', ''))
            data_point["value"] = value
        except (ValueError, IndexError):
            pass
    
    return data_point

def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract tables from PDF with better error handling
    """
    tables_data = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables from the page
                tables = page.extract_tables()
                
                for table_num, table in enumerate(tables, 1):
                    if not table or len(table) < 2:  # Skip empty or single-row tables
                        continue
                    
                    # Clean and normalize table data
                    cleaned_table = []
                    for row in table:
                        if not row:  # Skip empty rows
                            continue
                        # Clean each cell
                        cleaned_row = []
                        for cell in row:
                            if cell:
                                # Remove extra whitespace and normalize
                                cell = ' '.join(str(cell).split())
                            cleaned_row.append(cell if cell else "")
                        if any(cleaned_row):  # Only add non-empty rows
                            cleaned_table.append(cleaned_row)
                    
                    if len(cleaned_table) < 2:  # Skip tables with insufficient data
                        continue
                    
                    # Ensure all rows have the same number of columns
                    max_cols = max(len(row) for row in cleaned_table)
                    padded_table = [row + [""] * (max_cols - len(row)) for row in cleaned_table]
                    
                    # Store table data with metadata
                    table_data = {
                        "table_id": f"table_{page_num}_{table_num}",
                        "page": page_num,
                        "raw_data": padded_table,
                        "source": os.path.basename(pdf_path)
                    }
                    tables_data.append(table_data)
    except Exception as e:
        print(f"Error extracting tables: {str(e)}")
    
    return tables_data

def normalize_indian_numbers(text: str) -> str:
    """
    Enhanced number normalization for Indian financial formats
    """
    if not isinstance(text, str):
        return text
    
    # Remove currency symbols and whitespace
    text = text.replace('₹', '').replace('$', '').strip()
    
    # Handle negative numbers in parentheses
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
    
    # Check if it's a number with formatting
    if ',' in text or '.' in text:
        # Remove commas and spaces
        cleaned = text.replace(',', '').replace(' ', '')
        try:
            # Convert to float
            value = float(cleaned)
            return str(value)
        except ValueError:
            pass
    
    return text

def analyze_table_structure(table: List[List[str]]) -> Dict[str, Any]:
    """
    Analyze table structure and identify its type and components with better financial data handling
    """
    if not table:
        return {
            "type": "unknown",
            "headers": [],
            "data": [],
            "key_value_pairs": []
        }
    
    # Clean and standardize the table data
    cleaned_table = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
        if any(cleaned_row):  # Only add non-empty rows
            cleaned_table.append(cleaned_row)
    
    if not cleaned_table:
        return {
            "type": "unknown",
            "headers": [],
            "data": [],
            "key_value_pairs": []
        }
    
    # Try to identify table structure
    headers = cleaned_table[0]
    data_rows = cleaned_table[1:]
    
    # Check if first column contains financial line items
    has_line_items = any(any(keyword in row[0].lower() for keyword in [
        'expense', 'income', 'revenue', 'cost', 'profit', 'loss', 'total',
        'tax', 'depreciation', 'amortization', 'interest', 'travel'
    ]) for row in data_rows if row)
    
    key_value_pairs = []
    
    if has_line_items:
        # Process as financial statement table
        for row in data_rows:
            if not row or not row[0]:  # Skip empty rows
                continue
                
            line_item = row[0].strip()
            # Process each value column
            for i, value in enumerate(row[1:], 1):
                if value and i < len(headers):
                    period = headers[i].strip()
                    # Clean the value
                    cleaned_value = normalize_indian_numbers(value)
                    key_value_pairs.append({
                        "key": f"{line_item} ({period})",
                        "line_item": line_item,
                "period": period,
                        "value": cleaned_value,
                        "original_value": value
                    })
    
    return {
        "type": "financial" if has_line_items else "regular",
        "headers": headers,
        "data": data_rows,
        "key_value_pairs": key_value_pairs,
        "has_line_items": has_line_items
    }

def format_table_for_embedding(table: Dict[str, Any]) -> str:
    """
    Enhanced table formatting for better embedding and retrieval
    """
    text_parts = [f"Table on page {table['page']}:"]
    
    if table["type"] == "financial":
        text_parts.append("\nFinancial Statement Data:")
        # Group by line items
        line_items = {}
        for pair in table["key_value_pairs"]:
            item = pair["line_item"]
            if item not in line_items:
                line_items[item] = []
            line_items[item].append(f"{pair['period']}: {pair['original_value']}")
        
        # Format each line item with its values
        for item, values in line_items.items():
            text_parts.append(f"\n{item}:")
            for value in values:
                text_parts.append(f"  {value}")
    else:
        text_parts.append("\nHeaders:")
        text_parts.append(" | ".join(table["headers"]))
        text_parts.append("\nData:")
        for row in table["data"]:
            text_parts.append(" | ".join(str(cell) for cell in row))
    
    return "\n".join(text_parts)

# Load and process PDF documents
def load_pdf_documents(uploaded_files):
    """
    Enhanced PDF document loader with better table handling and error checking
    """
    uploaded_docs_folder = "./uploaded_doc"
    if not os.path.exists(uploaded_docs_folder):
        os.makedirs(uploaded_docs_folder)

    documents = []
    all_tables_data = []
    
    for uploaded_file in uploaded_files:
        pdf_filename = uploaded_file.name
        temp_pdf_path = os.path.join(uploaded_docs_folder, pdf_filename)
        
        # Save PDF temporarily
        with open(temp_pdf_path, "wb") as file:
            file.write(uploaded_file.getvalue())
        
        try:
            # Extract tables with enhanced processing
            tables_data = extract_tables_from_pdf(temp_pdf_path)
            
            if tables_data:  # Only process if tables were found
                all_tables_data.extend(tables_data)
                
                # Create document chunks for each table
                for table in tables_data:
                    if table.get("raw_data"):  # Only process tables with data
                        # Create a structured text representation of the table
                        table_text = f"Table from page {table['page']}:\n"
                        for row in table["raw_data"]:
                            table_text += " | ".join([str(cell) for cell in row]) + "\n"
                        
                        if table_text.strip():  # Only add if there's actual content
                            documents.append(Document(
                                page_content=table_text,
                                metadata={
                                    "file_name": pdf_filename,
                                    "file_type": "pdf",
                                    "content_type": "table",
                                    "table_id": table["table_id"],
                                    "page": table["page"]
                                }
                            ))
            
            # Extract and process regular text content
            with pdfplumber.open(temp_pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():  # Only add non-empty text
                        documents.append(Document(
                            page_content=text.strip(),
                            metadata={
                                "file_name": pdf_filename,
                                "file_type": "pdf",
                                "content_type": "text",
                                "page": page_num
                            }
                        ))
        
        except Exception as e:
            print(f"Error processing PDF {pdf_filename}: {str(e)}")
            # Try fallback method for text extraction
            try:
                loader = PyPDFLoader(temp_pdf_path)
                fallback_docs = loader.load()
                documents.extend(fallback_docs)
            except Exception as e:
                print(f"Fallback extraction failed for {pdf_filename}: {str(e)}")
        
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    # Ensure we have valid documents before processing
    if not documents:
        raise ValueError("No valid content could be extracted from the uploaded PDF(s)")

    # Process documents with error handling
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", ";", ",", " ", ""],
            length_function=len,
            add_start_index=True
        )

        processed_documents = []
        for doc in documents:
            if doc.page_content.strip():  # Only process non-empty documents
                splits = text_splitter.split_documents([doc])
                processed_documents.extend(splits)

        if not processed_documents:
            raise ValueError("No valid content remained after processing")

        # Create vector store
        vectorstore = FAISS.from_documents(
            processed_documents,
            embeddings
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 30,
                "lambda_mult": 0.7
            }
        )

        # Extract structured data
        extracted_data = parse_document_data(documents, all_tables_data)

        return {
            "vectorstore": vectorstore,
            "retriever": retriever,
            "extracted_data": extracted_data,
            "tables": all_tables_data
        }

    except Exception as e:
        print(f"Error in document processing: {str(e)}")
        raise

# Get session history
def get_session_history(session: str) -> ChatMessageHistory:
    """
    Get or create a new ChatMessageHistory for the session.
    
    Args:
        session (str): The session identifier
        
    Returns:
        ChatMessageHistory: The chat history object for the session
    """
    try:
        # Convert session to string if it's not already
        session_key = str(session) if not isinstance(session, str) else session
        
        # Initialize session store if needed
        if not hasattr(st.session_state, "chat_histories"):
            st.session_state.chat_histories = {}
        
        # Initialize chat history for this session if needed
        if session_key not in st.session_state.chat_histories:
            st.session_state.chat_histories[session_key] = ChatMessageHistory()
        
        return st.session_state.chat_histories[session_key]
        
    except Exception as e:
        print(f"Error in get_session_history: {str(e)}")
        # Return a new history instance as fallback
        return ChatMessageHistory()

# Create RAG chain for financial queries
def create_conversational_rag_chain(retriever):
    # Prompt for contextualizing questions based on chat history
    contextualize_q_system_prompt = """
    Given the chat history and latest user question about financial data:
    1. Identify the specific financial metrics being asked about
    2. Note any time periods or comparisons mentioned
    3. Formulate a clear, standalone question that captures all these elements
    Do NOT answer the question, just reformulate it to be specific and unambiguous.
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Main QA system prompt
    system_prompt = """
    You are a highly capable document analysis assistant. Your role is to provide accurate, contextual information from the documents you've analyzed.

    IMPORTANT GUIDELINES:

    1. DOCUMENT UNDERSTANDING:
       - Thoroughly analyze ALL provided context before answering
       - Pay special attention to numerical data, dates, and key metrics
       - Consider relationships between different sections and tables
       - Look for trends and patterns in the data

    2. ANSWER STRUCTURE:
       - Start with a clear, direct answer to the question
       - Use bullet points for listing multiple items
       - Present numerical data in tables when comparing multiple values
       - Use markdown formatting for better readability:
         * Headers: Use ### for sections
         * Lists: Use - or * for bullet points
         * Tables: Use proper markdown table format
         * Bold: Use ** for emphasis
         * Code/Numbers: Use ` for formatting

    3. DATA PRESENTATION:
       - Always include units with numerical values
       - Format large numbers with appropriate separators (e.g., 1,234,567)
       - Present percentages with consistent decimal places
       - When showing trends, indicate direction (increase/decrease)

    4. CONTEXT AND CITATIONS:
       - Cite specific pages or sections where information is found
       - Mention relevant time periods for temporal data
       - Explain any assumptions or limitations in the data
       - Provide context for technical terms or industry-specific concepts

    5. QUALITY CHECKS:
       - Verify numerical calculations
       - Cross-reference data points across different sections
       - Highlight any inconsistencies found
       - Indicate confidence level in the answer

    Remember to:
    - Be precise and factual
    - Use clear, professional language
    - Maintain consistent formatting
    - Provide complete, well-structured answers
    
    Context for this query:
    {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Create and combine chains
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    try:
        # Add message history handling
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
            history_factory_config=[
                {
                    "id": "default",
                    "history": ChatMessageHistory(),
                    "annotation": "chat_history"
                }
            ]
        )
        return conversational_rag_chain
    except Exception as e:
        st.error(f"Error initializing chat chain: {str(e)}")
        raise

def load_documents(uploaded_files):
    """
    Enhanced document loader for multiple file types
    """
    documents = []
    all_tables = []
    all_images = []
    processing_errors = []
    
    if not uploaded_files:
        raise ValueError("No files were uploaded")
    
    for uploaded_file in uploaded_files:
        try:
            file_type = uploaded_file.name.split('.')[-1].lower()
            st.info(f"Processing file: {uploaded_file.name} (Type: {file_type})")
            
            if file_type in ['pdf']:
                docs, tables = handle_pdf(uploaded_file)
                if docs:
                    documents.extend(docs)
                if tables:
                    all_tables.extend(tables)
                if not docs and not tables:
                    processing_errors.append(f"No content extracted from PDF: {uploaded_file.name}")
            
            elif file_type in ['xlsx', 'xls']:
                docs, tables = handle_excel(uploaded_file)
                if docs:
                    documents.extend(docs)
                if tables:
                    all_tables.extend(tables)
                if not docs and not tables:
                    processing_errors.append(f"No content extracted from Excel file: {uploaded_file.name}")
            
            elif file_type in ['ppt', 'pptx']:
                docs = handle_powerpoint(uploaded_file)
                if docs:
                    documents.extend(docs)
                else:
                    processing_errors.append(f"No content extracted from PowerPoint: {uploaded_file.name}")
            
            elif file_type in ['png', 'jpg', 'jpeg']:
                try:
                    docs, images = handle_image(uploaded_file)
                    if docs:
                        documents.extend(docs)
                    if images:
                        all_images.extend(images)
                    if not docs and not images:
                        processing_errors.append(f"No content extracted from image: {uploaded_file.name}")
                except RuntimeError as e:
                    processing_errors.append(f"Error processing image {uploaded_file.name}: {str(e)}")
                    # Still try to store the image even if OCR fails
                    try:
                        image = Image.open(uploaded_file)
                        all_images.append({
                            "image": image,
                            "text": "OCR processing failed",
                            "source": uploaded_file.name,
                            "type": "image"
                        })
                    except Exception as img_e:
                        processing_errors.append(f"Could not load image {uploaded_file.name}: {str(img_e)}")
            
            elif file_type in ['mp4', 'webm', 'avi', 'mov']:
                try:
                    # Save video to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process video
                    video_text = extract_video_text(tmp_file_path)
                    if video_text:
                        documents.append(Document(
                            page_content=video_text,
                            metadata={
                                "source": uploaded_file.name,
                                "type": "video",
                                "file": uploaded_file.name
                            }
                        ))
                        st.success(f"Successfully processed video: {uploaded_file.name}")
                    else:
                        processing_errors.append(f"No content extracted from video: {uploaded_file.name}")
                    
                    # Clean up
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
                        
                except Exception as e:
                    processing_errors.append(f"Error processing video {uploaded_file.name}: {str(e)}")
            
            elif file_type in ['zip', 'scorm']:
                docs = handle_scorm(uploaded_file)
                if docs:
                    documents.extend(docs)
                else:
                    processing_errors.append(f"No content extracted from SCORM package: {uploaded_file.name}")
            
            else:
                processing_errors.append(f"Unsupported file type: {file_type}")
            
        except Exception as e:
            error_msg = f"Error processing file {uploaded_file.name}: {str(e)}"
            st.error(error_msg)
            processing_errors.append(error_msg)
            continue

    # Check if we have any valid content
    has_content = bool(documents or all_images or all_tables)
    if not has_content:
        error_details = "\n".join(processing_errors) if processing_errors else "Unknown error"
        raise ValueError(f"No valid content could be extracted. Details:\n{error_details}")

    # Process documents if we have any
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", ";", ",", " ", ""],
            length_function=len,
            add_start_index=True
        )

        processed_documents = []
        for doc in documents:
            if doc.page_content.strip():
                splits = text_splitter.split_documents([doc])
                processed_documents.extend(splits)

        # Create vector store if we have processed documents
        if processed_documents:
            vectorstore = FAISS.from_documents(
                processed_documents,
                embeddings
            )
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 30,
                    "lambda_mult": 0.7
                }
            )
        else:
            # Create empty vectorstore for images-only case
            vectorstore = FAISS.from_texts(
                ["Image content present but no extractable text"],
                embeddings
            )
            retriever = vectorstore.as_retriever()

    else:
        # Create empty vectorstore for images-only case
        vectorstore = FAISS.from_texts(
            ["Image content present but no extractable text"],
            embeddings
        )
        retriever = vectorstore.as_retriever()

    # Extract structured data
    extracted_data = parse_document_data(documents, all_tables)

    return {
        "vectorstore": vectorstore,
        "retriever": retriever,
        "extracted_data": extracted_data,
        "tables": all_tables,
        "images": all_images,
        "processing_errors": processing_errors
    }
