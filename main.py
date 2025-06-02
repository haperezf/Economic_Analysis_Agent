# main.py

import argparse
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file before other imports
# to ensure configurations are available globally if needed by modules upon import.
load_dotenv()

# Assuming your project structure is something like:
# project_root/
#  ├── main.py
#  └── earia_agent/
#      └── agent_core/
#          └── earia_agent.py
# If main.py is outside a package containing earia_agent, you might need to adjust PYTHONPATH
# or use a different import strategy (e.g., if earia_agent is an installable package).
# For now, assuming earia_agent is in the Python path.
from earia_agent.agent_core.earia_agent import (
    EARIAAgent,
    ANALYSIS_TYPE_BASIC_RAG,
    ANALYSIS_TYPE_COT,
    ANALYSIS_TYPE_TOT,
    ANALYSIS_TYPE_AIN_EXTRACT
)

# Configure logging based on .env or default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s', # Added module and funcName
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Economic Agent for Regulatory Impact Analysis (EARIA)")
    parser.add_argument(
        "query",
        type=str,
        help="The query for economic impact analysis (e.g., 'Analyze CRC Resolution XYZ')."
    )
    parser.add_argument(
        "--doc_paths",
        type=str,
        nargs='+', # Allows multiple document paths
        default=None, # Default to None if not provided
        help="Path(s) to the regulatory document(s) to analyze. Separate multiple paths with spaces."
    )
    parser.add_argument(
        "--analysis_type",
        type=str,
        default=ANALYSIS_TYPE_BASIC_RAG,
        # Corrected: Provide a list of valid choices
        choices=[
            ANALYSIS_TYPE_BASIC_RAG,
            ANALYSIS_TYPE_COT,
            ANALYSIS_TYPE_TOT,
            ANALYSIS_TYPE_AIN_EXTRACT
        ],
        help=f"Type of analysis to perform (default: {ANALYSIS_TYPE_BASIC_RAG})."
    )
    parser.add_argument(
        "--aspect",
        type=str,
        default=None,
        help="Specific aspect to focus on for CoT analysis (e.g., 'consumer prices'). Required if analysis_type is 'cot_analysis'."
    )
    parser.add_argument(
        "--k_retrieval",
        type=int,
        default=5,
        help="Number of relevant document chunks to retrieve for context (default: 5)."
    )
    parser.add_argument(
        "--force_reprocess",
        action='store_true', # Makes it a flag, True if present
        help="Force reprocessing of documents even if they appear to be cached."
    )
    parser.add_argument(
        "--clear_kb",
        action='store_true',
        help="Clear the entire knowledge base before starting. USE WITH CAUTION."
    )

    args = parser.parse_args()

    if args.analysis_type == ANALYSIS_TYPE_COT and not args.aspect:
        parser.error("--aspect is required when analysis_type is 'cot_analysis'.")
        # parser.error() exits, so a return here is mostly for explicitness or if error behavior changes.

    logger.info("Starting EARIA agent with provided arguments...")
    logger.debug(f"Parsed arguments: {args}")

    try:
        agent = EARIAAgent() # Assumes default configurations for underlying handlers are okay

        if args.clear_kb:
            confirm = input("Are you sure you want to clear the entire knowledge base? This cannot be undone. (yes/no): ")
            if confirm.lower() == 'yes':
                logger.info("User confirmed. Clearing knowledge base...")
                agent.clear_knowledge_base()
                logger.info("Knowledge base cleared successfully.")
            else:
                logger.info("Knowledge base clearing aborted by user.")
                return # Exit if clearing was intended but aborted

        # Check knowledge base status if no new documents are explicitly provided for processing
        # Use the corrected get_collection_count() method
        # This check assumes agent.vector_manager is accessible and has get_collection_count()
        try:
            current_kb_count = agent.vector_manager.get_collection_count()
            if not args.doc_paths and current_kb_count == 0:
                logger.warning("No document paths provided for this session, and the knowledge base is currently empty. "
                               "Analysis might be very limited or rely solely on the LLM's general knowledge if no context is retrieved.")
            elif args.doc_paths:
                 logger.info(f"Document paths provided: {args.doc_paths}. These will be processed if new or --force_reprocess is set.")
            else:
                 logger.info(f"No document paths provided for this session. Using existing knowledge base with {current_kb_count} items.")

        except AttributeError as ae:
            logger.error(f"Could not access vector_manager or get_collection_count: {ae}. "
                         "The EARIAAgent might not be fully initialized or the method is missing.")
            # Depending on severity, you might want to exit here.


        logger.info(f"Performing analysis of type '{args.analysis_type}' for query: '{args.query}'")
        analysis_result = agent.analyze_economic_impact(
            query=args.query,
            # Corrected: Pass args.doc_paths directly. It will be None if not provided.
            document_paths=args.doc_paths,
            analysis_type=args.analysis_type,
            k_retrieval=args.k_retrieval,
            aspect_to_analyze=args.aspect,
            force_reprocess_docs=args.force_reprocess
        )

        print("\n--- EARIA Analysis Result ---")
        print(analysis_result)
        print("---------------------------\n")

    except Exception as e:
        logger.critical(f"A critical error occurred during agent operation: {e}", exc_info=True)
        print(f"\nAn critical error occurred. Please check the logs for details. Error: {e}")

if __name__ == "__main__":
    main()