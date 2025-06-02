# earia_agent/data_ingestion/text_processor.py

import re
from typing import List, Optional # Optional might be needed if separators can be None
from langchain_core.documents import Document # Corrected import path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import os

logger = logging.getLogger(__name__)
# Configure logging at the application entry point if this is part of a larger app.
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

class TextProcessor:
    """
    Handles cleaning and chunking of text content from documents.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans the input text by removing excessive whitespace and normalizing characters.

        Args:
            text (str): The raw text to clean.

        Returns:
            str: The cleaned text.
        """
        if not text:
            return ""
        
        # 1. Normalize specific Unicode whitespace characters (e.g., non-breaking space, zero-width space) to a standard space.
        # \u00A0 = No-Break Space
        # \u200B = Zero Width Space
        # \u200C = Zero Width Non-Joiner
        # \u200D = Zero Width Joiner
        # \uFEFF = Zero Width No-Break Space (often a BOM character)
        text = re.sub(r'[\u00A0\u200B-\u200D\uFEFF]+', ' ', text)
        
        # 2. Remove multiple spaces and newlines, then strip leading/trailing whitespace.
        # \s+ matches any whitespace character (space, tab, newline, etc.) one or more times.
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Optional: Convert to lowercase, but consider if case is important for legal/economic terms.
        # For this agent, preserving case might be better for named entities and acronyms.
        # text = text.lower()

        # Further cleaning steps can be added here, e.g.:
        # - Removing specific boilerplate if identifiable by patterns
        # - Normalizing hyphens or quotes (e.g., re.sub(r'[“”]', '"', text))
        # - Handling ligatures if they cause issues

        # logger.debug(f"Cleaned text (first 100 chars): {text[:100]}") # Debug logging can be verbose
        return text

    @staticmethod
    def chunk_documents(
        documents: List[Document], # Corrected type hint
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None # Corrected type hint for optional separators
    ) -> List[Document]: # Corrected type hint
        """
        Splits a list of LangChain Documents into smaller chunks after cleaning their content.

        Args:
            documents (List[Document]): The list of documents to chunk.
            chunk_size (int): The maximum size of each chunk (in characters).
            chunk_overlap (int): The number of characters to overlap between chunks.
            separators (Optional[List[str]]): Custom separators for splitting.
                                            Defaults to LangChain's RecursiveCharacterTextSplitter defaults.

        Returns:
            List[Document]: A list of chunked LangChain Document objects. Returns empty list if no documents.
        """
        if not documents:
            logger.warning("No documents provided for chunking.")
            return [] # Corrected: Return empty list

        if separators is None:
            # Default separators from RecursiveCharacterTextSplitter are often good for general text.
            # These are ["\n\n", "\n", " ", ""].
            # The user's provided list is more extensive. Using it:
            # Note: Using very granular separators like single punctuation with a large chunk_size
            # might not have much effect if broader separators (like "\n\n") are present.
            # If broader separators are absent, then these fine-grained ones will be used.
            separators = ["\n\n", "\n", ". ", "! ", "? ", " ", "", ".", ",", ";", ":", "(", ")", "[", "]", "{", "}"]
            # A slightly more conservative but still enhanced list could be:
            # separators = ["\n\n", "\n", ". ", "።", "。", "！", "？", "｡", "\r", "\t", " ", ""] # Includes some CJK punctuation

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False, # Treat separators as literals
            separators=separators
        )
        
        all_chunks: List[Document] = [] # Corrected: Initialize as empty list with type hint
        for doc_idx, doc in enumerate(documents):
            if not hasattr(doc, 'page_content') or not doc.page_content or not doc.page_content.strip():
                source_info = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown (no metadata)'
                logger.warning(f"Document {doc_idx} from source '{source_info}' has no content or is whitespace only, skipping.")
                continue

            cleaned_content = TextProcessor.clean_text(doc.page_content)
            if not cleaned_content: # Check if cleaning resulted in empty content
                source_info = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown (no metadata)'
                logger.warning(f"Document {doc_idx} from source '{source_info}' resulted in empty content after cleaning, skipping.")
                continue
            
            # Option 1 chosen by user: Split text then create new Document objects for chunks
            text_chunks = text_splitter.split_text(cleaned_content)
            
            parent_metadata = doc.metadata if hasattr(doc, 'metadata') else {}

            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = parent_metadata.copy() # Inherit metadata from parent document
                # Create a robust chunk_id, handling cases where 'source' might be missing
                source_name = parent_metadata.get('source', f'doc{doc_idx}')
                chunk_metadata["chunk_id"] = f"{source_name}_page_{doc_idx}_chunk_{i}"
                
                # If original doc had page numbers (e.g. from PDF loader), try to preserve or estimate
                if "page" in parent_metadata: # Langchain PDFLoader often uses 'page' (0-indexed)
                    chunk_metadata["original_page"] = parent_metadata["page"] + 1 # User-friendly 1-indexed

                new_chunk_doc = Document(page_content=chunk_text, metadata=chunk_metadata)
                all_chunks.append(new_chunk_doc)

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks.")
        return all_chunks

if __name__ == '__main__':
    # Example Usage
    doc_content1 = """
    Artículo 1: Objeto de la regulación.     Con muchos espacios. Y\u00A0non-breaking\u00A0spaces.
    El presente reglamento tiene por objeto establecer las condiciones bajo las cuales los operadores de telecomunicaciones...
    Este es un párrafo largo con mucha información importante que necesita ser dividida en partes más pequeñas.
    La economía digital en Colombia ha crecido exponencialmente.

    Artículo 2: Definiciones.
    Para efectos de esta resolución, se entenderá por 'impacto económico' cualquier alteración significativa...
    Otro párrafo extenso que detalla múltiples conceptos y sus interrelaciones.
    """
    doc_content2 = "Un documento corto. Sin mucho más que decir.\n\n\nDemasiados saltos."
    doc_content_empty = "   \n \t \u200B   " # Whitespace and zero-width space
    doc_content_none = None


    # Corrected: Initialize sample_docs with Document objects
    sample_docs_data = [
        Document(page_content=doc_content1, metadata={"source": "Resolucion_CRC_123.pdf", "category": "regulation"}),
        Document(page_content=doc_content2, metadata={"source": "Nota_Tecnica_005.txt", "category": "technical_note"}),
        Document(page_content=doc_content_empty, metadata={"source": "Emptyish_Doc.txt"}),
        Document(page_content=doc_content_none, metadata={"source": "None_Content_Doc.txt"}) # Test with None content
    ]

    # Test cleaning
    logger.info("--- Testing clean_text ---")
    test_text_for_cleaning = "  Mucho \t espacio \n\n y \u00A0 non-breaking \u200B space.  "
    cleaned_version = TextProcessor.clean_text(test_text_for_cleaning)
    logger.info(f"Original for cleaning: '{test_text_for_cleaning}'")
    logger.info(f"Cleaned version: '{cleaned_version}'")
    logger.info(f"Cleaned empty string: '{TextProcessor.clean_text('')}'")
    logger.info(f"Cleaned content_empty: '{TextProcessor.clean_text(doc_content_empty)}'")


    # Test chunking
    logger.info("\n--- Testing chunk_documents ---")
    # Filter out documents that might be problematic before passing to chunk_documents if they aren't Document instances
    valid_sample_docs = [doc for doc in sample_docs_data if isinstance(doc, Document) and doc.page_content is not None]

    chunked_documents = TextProcessor.chunk_documents(valid_sample_docs, chunk_size=100, chunk_overlap=20)
    logger.info(f"Number of original valid documents: {len(valid_sample_docs)}")
    logger.info(f"Number of chunks generated: {len(chunked_documents)}")
    for i, chunk in enumerate(chunked_documents):
        logger.info(f"Chunk {i+1} (Source: {chunk.metadata.get('source')}, Chunk ID: {chunk.metadata.get('chunk_id')}, Page: {chunk.metadata.get('original_page', 'N/A')}):")
        logger.info(f"'{chunk.page_content}'")
        logger.info("---")

    logger.info("\n--- Testing chunk_documents with empty list ---")
    empty_chunks = TextProcessor.chunk_documents([])
    logger.info(f"Number of chunks from empty list: {len(empty_chunks)}")