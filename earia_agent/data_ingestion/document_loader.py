# earia_agent/data_ingestion/document_loader.py

import os
import fnmatch # Moved import to top
from typing import List, Optional # Changed from Dict, Union as they weren't used in return types
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    # DirectoryLoader, # Not used in the final corrected version of load_from_directory
    UnstructuredFileLoader # Fallback for other types
)
from langchain_core.documents import Document # Corrected import path from langchain.docstore.document
import logging

logger = logging.getLogger(__name__)
# Configure logging at the application entry point if this is part of a larger app.
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

class DocumentLoader:
    """
    Handles loading of documents from various file formats.
    """

    @staticmethod
    def load_single_document(file_path: str) -> List[Document]: # Corrected type hint
        """
        Loads a single document based on its file extension.

        Args:
            file_path (str): The path to the document file.

        Returns:
            List[Document]: A list of LangChain Document objects.
                            Returns an empty list if loading fails, file not found, or format is unsupported.
        """
        logger.info(f"Attempting to load document: {file_path}")
        try:
            if not os.path.exists(file_path) or not os.path.isfile(file_path): # Also check if it's a file
                logger.error(f"File not found or is not a file: {file_path}")
                return [] # Corrected: Return empty list

            # Corrected: Get the extension (second part of tuple) and then lower()
            file_extension = os.path.splitext(file_path)[1].lower()
            loader = None # Initialize loader

            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension in [".html", ".htm"]:
                loader = UnstructuredHTMLLoader(file_path)
            else:
                logger.info(f"No specific loader for extension '{file_extension}'. Attempting generic UnstructuredFileLoader for {file_path}.")
                # UnstructuredFileLoader can often handle various types like .csv, .xls, .json, .md etc.
                # mode="elements" or mode="single" can be chosen based on desired output granularity
                loader = UnstructuredFileLoader(file_path, mode="elements", strategy="auto")

            if loader is None: # Should not happen if UnstructuredFileLoader is the fallback, but as a safeguard
                logger.warning(f"Could not determine a loader for {file_path} with extension '{file_extension}'.")
                return []

            documents = loader.load()
            if not documents:
                logger.warning(f"Loader for {file_path} returned no documents.")
                return []
                
            logger.info(f"Successfully loaded {len(documents)} document(s)/page(s) from {file_path}")
            
            # Add source metadata consistently
            file_name = os.path.basename(file_path)
            for doc in documents:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {} # Ensure metadata attribute exists
                doc.metadata["source"] = file_name
                # You might want to add the full file_path as well or instead
                # doc.metadata["file_path"] = file_path
            return documents
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}", exc_info=True)
            return [] # Corrected: Return empty list

    @staticmethod
    def _matches_glob(file_path: str, base_dir: str, glob_pattern: str) -> bool:
        """
        Helper to check if a file_path matches a glob pattern.
        The pattern is matched against the path relative to base_dir.
        """
        if glob_pattern == "**/*" or glob_pattern == "**/*.*": # Common "match all" patterns
            return True
        
        relative_path = os.path.relpath(file_path, base_dir)
        # Normalize path separators for consistent matching, especially on Windows
        normalized_relative_path = relative_path.replace("\\", "/")
        normalized_glob_pattern = glob_pattern.replace("\\", "/")
        
        return fnmatch.fnmatch(normalized_relative_path, normalized_glob_pattern)

    @staticmethod
    def load_from_directory(directory_path: str, glob_pattern: str = "**/*.*") -> List[Document]: # Corrected type hint
        """
        Loads all documents from a specified directory that match the glob pattern.

        Args:
            directory_path (str): The path to the directory containing documents.
            glob_pattern (str): Glob pattern to match files (e.g., "*.pdf", "docs/**/*.txt").
                                Defaults to match all files in all subdirectories.

        Returns:
            List[Document]: A list of all loaded LangChain Document objects.
        """
        logger.info(f"Attempting to load documents from directory: {directory_path} with pattern: {glob_pattern}")
        if not os.path.isdir(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return [] # Corrected: Return empty list

        all_documents: List[Document] = [] # Corrected: Initialize as empty list with type hint
        
        # Alternative using pathlib for cleaner globbing (recommended for Python 3.4+):
        # from pathlib import Path
        # base_path = Path(directory_path)
        # for file_path_obj in base_path.rglob(glob_pattern): # rglob handles recursive "**/"" patterns naturally
        #     if file_path_obj.is_file():
        #         logger.debug(f"Found by rglob: {file_path_obj}")
        #         docs = DocumentLoader.load_single_document(str(file_path_obj))
        #         if docs: # Ensure docs is not None or empty
        #             all_documents.extend(docs)
        # logger.info(f"Finished pathlib rglob. Total documents: {len(all_documents)}")


        # Using os.walk and custom _matches_glob as per original structure
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if DocumentLoader._matches_glob(file_path, directory_path, glob_pattern):
                    logger.debug(f"File matched glob: {file_path}")
                    docs = DocumentLoader.load_single_document(file_path)
                    if docs: # Ensure docs is not None or empty
                        all_documents.extend(docs)
                # else: # Optional: log files that don't match
                #     logger.debug(f"File did not match glob: {file_path}")
        
        logger.info(f"Successfully loaded {len(all_documents)} document(s)/page(s) from directory {directory_path}")
        return all_documents


if __name__ == '__main__':
    example_docs_path = "example_docs_loader"
    os.makedirs(example_docs_path, exist_ok=True)
    
    sample_txt_path = os.path.join(example_docs_path, "sample.txt")
    sample_md_path = os.path.join(example_docs_path, "sample.md")
    subdir_path = os.path.join(example_docs_path, "subdir")
    os.makedirs(subdir_path, exist_ok=True)
    nested_txt_path = os.path.join(subdir_path, "nested.txt")

    if not os.path.exists(sample_txt_path):
        with open(sample_txt_path, "w", encoding="utf-8") as f:
            f.write("This is a sample text document for testing the document loader.")
    
    if not os.path.exists(sample_md_path):
        with open(sample_md_path, "w", encoding="utf-8") as f:
            f.write("# Sample Markdown\nThis is a test markdown file.")

    if not os.path.exists(nested_txt_path):
        with open(nested_txt_path, "w", encoding="utf-8") as f:
            f.write("This is a nested text document.")

    # Test loading a single document
    logger.info(f"\n--- Testing load_single_document ('{sample_txt_path}') ---")
    loaded_txt_docs = DocumentLoader.load_single_document(sample_txt_path)
    if loaded_txt_docs: # Check if the list is not empty
        logger.info(f"Number of documents/pages from TXT: {len(loaded_txt_docs)}")
        # Corrected: Access elements of the list
        logger.info(f"Content of first loaded TXT doc: {loaded_txt_docs[0].page_content[:100]}...")
        logger.info(f"Metadata: {loaded_txt_docs[0].metadata}")
    else:
        logger.warning(f"No documents loaded from {sample_txt_path}")

    logger.info(f"\n--- Testing load_single_document with unknown extension (Markdown) ('{sample_md_path}') ---")
    loaded_md_docs = DocumentLoader.load_single_document(sample_md_path)
    if loaded_md_docs:
        logger.info(f"Number of documents/pages from MD: {len(loaded_md_docs)}")
        logger.info(f"Content of first loaded MD doc: {loaded_md_docs[0].page_content[:100]}...")
        logger.info(f"Metadata: {loaded_md_docs[0].metadata}")
    else:
        logger.warning(f"No documents loaded from {sample_md_path}")


    # Test loading from a directory
    logger.info(f"\n--- Testing load_from_directory ('{example_docs_path}', glob='**/*.txt') ---")
    # This glob should pick up sample.txt and subdir/nested.txt
    all_dir_txt_docs = DocumentLoader.load_from_directory(example_docs_path, glob_pattern="**/*.txt")
    logger.info(f"Total .txt documents loaded from directory: {len(all_dir_txt_docs)}")
    if all_dir_txt_docs:
        for i, doc in enumerate(all_dir_txt_docs):
            logger.info(f"  Doc {i+1} Source: {doc.metadata.get('source', 'N/A')}, Content: {doc.page_content[:50]}...")
    else:
        logger.warning(f"No .txt documents loaded from directory {example_docs_path}")

    logger.info(f"\n--- Testing load_from_directory ('{example_docs_path}', glob='*.md') ---")
    # This glob should pick up sample.md from the root of example_docs_path
    all_dir_md_docs = DocumentLoader.load_from_directory(example_docs_path, glob_pattern="*.md")
    logger.info(f"Total .md documents loaded from directory: {len(all_dir_md_docs)}")
    if all_dir_md_docs:
        logger.info(f"Content of first .md doc from directory: {all_dir_md_docs[0].page_content[:100]}...")
        logger.info(f"Metadata of first .md doc from directory: {all_dir_md_docs[0].metadata}")
    else:
        logger.warning(f"No .md documents loaded from directory {example_docs_path} matching *.md")
        
    logger.info(f"\n--- Testing load_from_directory ('{example_docs_path}', glob='**/*.*') ---")
    # This glob should pick up all files
    all_dir_docs_all_types = DocumentLoader.load_from_directory(example_docs_path, glob_pattern="**/*.*")
    logger.info(f"Total documents (all types) loaded from directory: {len(all_dir_docs_all_types)}")
    if all_dir_docs_all_types:
         for i, doc in enumerate(all_dir_docs_all_types):
            logger.info(f"  Doc {i+1} Source: {doc.metadata.get('source', 'N/A')}, Content: {doc.page_content[:50]}...")
    else:
        logger.warning(f"No documents (all types) loaded from directory {example_docs_path}")


    # Clean up dummy directory (optional)
    # import shutil
    # logger.info(f"\n--- Cleaning up '{example_docs_path}' ---")
    # try:
    #     shutil.rmtree(example_docs_path)
    #     logger.info("Cleanup successful.")
    # except OSError as e:
    #     logger.error(f"Error during cleanup: {e}")