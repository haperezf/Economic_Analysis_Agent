# earia_agent/agent_core/earia_agent.py

import os
import hashlib
from typing import List, Dict, Any, Optional

from ..data_ingestion.document_loader import DocumentLoader
from ..data_ingestion.text_processor import TextProcessor
from ..knowledge_base.vector_store_manager import VectorStoreManager
from .llm_handler import LLMHandler
import logging

logger = logging.getLogger(__name__)

# Define analysis types
ANALYSIS_TYPE_BASIC_RAG = "basic_rag"
ANALYSIS_TYPE_COT = "cot_analysis"
ANALYSIS_TYPE_TOT = "tot_exploration"
ANALYSIS_TYPE_AIN_EXTRACT = "ain_extraction"

# Directorio por defecto para los documentos del agente, puede ser sobrescrito por .env
DEFAULT_DOCS_DIRECTORY = os.getenv("EARIA_DOCUMENTS_DIR", "./documents")

class EARIAAgent:
    def __init__(self,
                 llm_handler: Optional[LLMHandler] = None,
                 vector_store_manager: Optional[VectorStoreManager] = None,
                 document_loader: Optional[DocumentLoader] = None,
                 text_processor: Optional[TextProcessor] = None,
                 documents_directory: str = DEFAULT_DOCS_DIRECTORY):
        logger.info("Initializing EARIAAgent...")
        self.doc_loader = document_loader if document_loader else DocumentLoader()
        self.text_processor = text_processor if text_processor else TextProcessor()
        self.vector_manager = vector_store_manager if vector_store_manager else VectorStoreManager()
        # LLMHandler usará los prompts actualizados y su propia config de modelo (de .env)
        self.llm_handler = llm_handler if llm_handler else LLMHandler()
        
        self.documents_directory = documents_directory
        logger.info(f"EARIAAgent utilizará el directorio de documentos: {os.path.abspath(self.documents_directory)}")
        
        # El caché de _processed_files_cache es útil si se procesan archivos individuales con doc_paths.
        # Para indexación de directorio completo, el force_reprocess es el control principal.
        self._processed_files_cache = set() 
        self._directory_indexed_this_session = False # Para evitar reindexar innecesariamente por defecto
        logger.info("EARIAAgent initialized.")

    def _get_file_hash(self, file_path: str) -> Optional[str]:
        # (Este método se mantiene igual que en tu última versión funcional)
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except FileNotFoundError:
            logger.error(f"File not found for hashing: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}", exc_info=True)
            return None

    def _ensure_directory_indexed(self, force_reprocess_all: bool = False) -> bool:
        """
        Asegura que todos los documentos en self.documents_directory estén indexados.
        Usa un flag simple para evitar reindexar en la misma sesión a menos que se fuerce.
        Para un control más granular de archivos individuales, el _processed_files_cache
        se usaría si se pasaran listas de archivos.
        """
        if not force_reprocess_all and self._directory_indexed_this_session:
            logger.info(f"El directorio '{self.documents_directory}' ya fue indexado en esta sesión. Saltando.")
            return True

        logger.info(f"Asegurando indexación del directorio: {self.documents_directory} (force_reprocess_all={force_reprocess_all})")
        if not os.path.isdir(self.documents_directory):
            logger.error(f"El directorio de documentos configurado no existe: {self.documents_directory}")
            return False

        # Usar DocumentLoader.load_from_directory
        # Asumimos que DocumentLoader tiene un método load_from_directory que devuelve List[Document]
        # y maneja diferentes tipos de archivo usando un glob_pattern como "**/*.*" por defecto.
        try:
            # El DocumentLoader.load_from_directory que te proporcioné antes ya hace os.walk
            # y llama a load_single_document internamente.
            # Si tu DocumentLoader.load_from_directory es diferente, ajusta esto.
            # Por ahora, asumo que devuelve una lista plana de todos los LangChain Documents.
            
            # Para simplificar, vamos a obtener la lista de archivos y procesarlos uno por uno
            # si load_from_directory no es el método preferido para procesar todo el directorio.
            # La implementación de load_from_directory que te di antes ya hace un bucle
            # y llama a load_single_document.
            
            # Si forzamos el reprocesamiento de todo, limpiamos el caché de archivos individuales.
            if force_reprocess_all:
                self._processed_files_cache.clear()
                logger.info("Caché de archivos procesados limpiado debido a force_reprocess_all.")

            # Recolectar todos los paths de archivos para procesar con la lógica existente
            all_file_paths_in_dir = []
            for root, _, files in os.walk(self.documents_directory):
                for file in files:
                    # Aquí podrías añadir un filtro de extensiones si lo deseas
                    # ej. if file.lower().endswith(('.pdf', '.txt', '.docx')):
                    all_file_paths_in_dir.append(os.path.join(root, file))
            
            if not all_file_paths_in_dir:
                logger.warning(f"No se encontraron archivos en el directorio: {self.documents_directory}")
                self._directory_indexed_this_session = True # Marcamos como "hecho" para no reintentar
                return True # No es un error si el directorio está vacío

            logger.info(f"Se encontraron {len(all_file_paths_in_dir)} archivos para posible indexación en '{self.documents_directory}'.")
            
            # Usamos el método process_and_index_documents existente que maneja caché por archivo
            # y force_reprocess para archivos individuales si la caché no se limpió globalmente.
            success = self.process_and_index_documents(all_file_paths_in_dir, force_reprocess=force_reprocess_all)
            
            if success:
                self._directory_indexed_this_session = True
                logger.info(f"Indexación/verificación del directorio '{self.documents_directory}' completada exitosamente.")
            else:
                logger.error(f"Falló la indexación/verificación de uno o más archivos en el directorio '{self.documents_directory}'.")
            return success

        except Exception as e:
            logger.error(f"Error crítico durante la indexación del directorio {self.documents_directory}: {e}", exc_info=True)
            return False


    def process_and_index_documents(self, doc_paths_to_process: List[str], force_reprocess: bool = False) -> bool:
        """
        Procesa una lista específica de rutas de documentos.
        force_reprocess aquí se aplica a estos archivos específicos.
        Este método es llamado por _ensure_directory_indexed.
        """
        any_new_documents_added_successfully = False
        all_input_docs_handled_without_critical_failure = True

        for doc_path in doc_paths_to_process:
            if not os.path.exists(doc_path) or not os.path.isfile(doc_path):
                logger.warning(f"Ruta de documento no existe o no es un archivo, saltando: {doc_path}")
                continue 

            file_hash = self._get_file_hash(doc_path)
            if file_hash is None:
                logger.warning(f"Saltando documento por error de hash: {doc_path}")
                all_input_docs_handled_without_critical_failure = False
                continue
                
            cache_key = (doc_path, file_hash)

            if not force_reprocess and cache_key in self._processed_files_cache:
                logger.debug(f"Documento {doc_path} (hash: {file_hash}) ya procesado y en caché. Saltando.")
                continue
            
            logger.info(f"Procesando para indexar: {doc_path}")
            try:
                loaded_pages = self.doc_loader.load_single_document(doc_path)
                if not loaded_pages:
                    logger.warning(f"No se cargó contenido de {doc_path}. Saltando.")
                    continue

                chunks = self.text_processor.chunk_documents(loaded_pages)
                if not chunks:
                    logger.warning(f"No se generaron chunks de {doc_path}. Saltando.")
                    continue
                
                if self.vector_manager.add_documents(chunks): # Asumiendo que add_documents devuelve bool
                    self._processed_files_cache.add(cache_key)
                    any_new_documents_added_successfully = True
                    logger.info(f"Procesado e indexado exitosamente: {doc_path}")
                else:
                    logger.error(f"Falló al añadir chunks al vector store para {doc_path}.")
                    all_input_docs_handled_without_critical_failure = False
            
            except Exception as e:
                logger.error(f"Error crítico procesando/indexando documento {doc_path}: {e}", exc_info=True)
                all_input_docs_handled_without_critical_failure = False
        
        if any_new_documents_added_successfully:
            try:
                count = self.vector_manager.get_collection_count()
                logger.info(f"Colección '{self.vector_manager.collection_name}' ahora tiene {count} items.")
            except Exception as e:
                logger.warning(f"Error obteniendo conteo de colección después de procesar: {e}")
        
        return all_input_docs_handled_without_critical_failure

    def analyze_economic_impact(
        self,
        query: str,
        # document_paths ya no es un input principal para este método si indexamos todo el dir
        # Mantenido como opcional si se quiere FORZAR el procesamiento de archivos específicos en esta llamada
        document_paths_to_ensure: Optional[List[str]] = None, 
        analysis_type: str = ANALYSIS_TYPE_BASIC_RAG,
        k_retrieval: int = 7, # Default k aumentado para análisis más profundos
        aspect_to_analyze: Optional[str] = None,
        force_reprocess_docs: bool = False # Esto forzaría el reprocesamiento del directorio completo
    ) -> str:
        logger.info(f"Iniciando análisis económico. Consulta: '{query}', Tipo: {analysis_type}, Aspecto/Tema Principal: '{aspect_to_analyze}'")

        # Asegurar que el directorio de conocimiento base esté indexado
        # force_reprocess_docs aquí se refiere a forzar el re-procesamiento de todo el directorio
        if not self._ensure_directory_indexed(force_reprocess_all=force_reprocess_docs):
            # Si la indexación del directorio falla críticamente, podríamos no poder continuar
             return ("Error Crítico: Falló la indexación del directorio de documentos base. "
                    "El análisis no puede proceder de manera confiable.")

        # Si se especificaron document_paths_to_ensure, procesarlos específicamente también.
        # Esto es útil si se quiere asegurar que ciertos archivos nuevos (no en el dir principal)
        # o actualizados recientemente se consideren inmediatamente.
        if document_paths_to_ensure:
            logger.info(f"Asegurando procesamiento adicional para documentos específicos: {document_paths_to_ensure}")
            if not self.process_and_index_documents(document_paths_to_ensure, force_reprocess=True): # Forzar para estos
                 return ("Error Crítico: Falló el procesamiento de uno o más documentos especificados adicionalmente. "
                        "El análisis no puede proceder.")


        collection_count = self.vector_manager.get_collection_count()
        
        retrieved_chunks_for_analysis: List[Dict[str, Any]] = []
        combined_chunks_text_for_summary = "Contexto documental no disponible o no recuperado."
        query_for_retrieval = query

        if not query and analysis_type not in [ANALYSIS_TYPE_AIN_EXTRACT]:
            logger.warning(f"No se proporcionó una consulta y el tipo de análisis '{analysis_type}' podría requerirla. El análisis será limitado.")
        
        elif query: 
            if analysis_type == ANALYSIS_TYPE_AIN_EXTRACT and document_paths_to_ensure and len(document_paths_to_ensure) == 1:
                # Para AIN, si se dio un doc específico en document_paths_to_ensure, enfocar la query ahí.
                ain_doc_name = os.path.basename(document_paths_to_ensure[0])
                query_for_retrieval = f"Extraer componentes del AIN del documento {ain_doc_name} y contexto relevante."
            
            if collection_count > 0:
                logger.info(f"Recuperando contexto relevante de la base de conocimiento para: '{query_for_retrieval}' (k={k_retrieval})...")
                retrieved_docs_objects = self.vector_manager.similarity_search(query_for_retrieval, k=k_retrieval)
                retrieved_chunks_for_analysis = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs_objects if doc.page_content]
                
                if retrieved_chunks_for_analysis:
                    combined_chunks_text_for_summary = "\n\n---\n\n".join([chunk["page_content"] for chunk in retrieved_chunks_for_analysis])
                    logger.info(f"Contexto recuperado para análisis (longitud combinada: {len(combined_chunks_text_for_summary)} caracteres).")
                    if not combined_chunks_text_for_summary.strip():
                        combined_chunks_text_for_summary = "Contexto recuperado pero el contenido combinado es vacío."
                        logger.warning(combined_chunks_text_for_summary)
                else:
                    logger.warning(f"No se encontró contexto relevante para: '{query_for_retrieval}'.")
            else:
                logger.warning("La base de conocimiento vectorial está vacía. No se puede recuperar contexto.")
        
        logger.info("Generando título y resumen inicial del contexto...")
        title_and_initial_summary = self.llm_handler.generate_title_and_summary(
            combined_context_text=combined_chunks_text_for_summary,
            original_query=query
        )

        main_analysis_response = ""
        if analysis_type == ANALYSIS_TYPE_BASIC_RAG:
            main_analysis_response = self.llm_handler.generate_response_rag(query, retrieved_chunks_for_analysis)
        elif analysis_type == ANALYSIS_TYPE_COT:
            effective_aspect = aspect_to_analyze if aspect_to_analyze else query 
            if not effective_aspect: # Si tanto aspect como query son vacíos
                logger.error("Análisis CoT requiere una query o un 'aspect_to_analyze'.")
                error_msg = "Error: Se requiere una consulta o un tema principal ('aspect_to_analyze') para el análisis CoT."
                return f"{title_and_initial_summary}\n\n--- Análisis Económico Profundo (CoT) ---\n{error_msg}"
            main_analysis_response = self.llm_handler.generate_response_cot_analysis(query, retrieved_chunks_for_analysis, effective_aspect)
        elif analysis_type == ANALYSIS_TYPE_TOT:
            main_analysis_response = self.llm_handler.generate_response_tot_exploration(query, retrieved_chunks_for_analysis)
        elif analysis_type == ANALYSIS_TYPE_AIN_EXTRACT:
            main_analysis_response = self.llm_handler.generate_response_extract_ain(retrieved_chunks_for_analysis, query_for_ain=query)
        else:
            error_msg = f"Error: Tipo de análisis desconocido '{analysis_type}'."
            return f"{title_and_initial_summary}\n\n--- Análisis (Tipo Desconocido) ---\n{error_msg}"

        analysis_title_str = analysis_type.replace('_', ' ').title()
        if analysis_type == ANALYSIS_TYPE_COT and aspect_to_analyze:
            aspect_snippet = aspect_to_analyze[:50] + "..." if len(aspect_to_analyze) > 50 else aspect_to_analyze
            analysis_title_str = f"Análisis Económico Profundo (CoT) sobre: {aspect_snippet.capitalize()}"
        elif analysis_type == ANALYSIS_TYPE_BASIC_RAG:
            analysis_title_str = "Respuesta Basada en Contexto Documental (RAG)" # Generalizado
        # ... (otros títulos pueden generalizarse similarmente) ...
            
        final_response = (
            f"{title_and_initial_summary}\n\n\n"
            f"--- {analysis_title_str} ---\n"
            f"{main_analysis_response}"
        )
        
        return final_response.strip()

    def clear_knowledge_base(self):
        logger.info("Clearing knowledge base...")
        try:
            self.vector_manager.clear_collection() # Asume que esto borra todo en la colección
            self._processed_files_cache.clear()
            self._directory_indexed_this_session = False # Resetear flag de indexación
            logger.info("Knowledge base cleared and re-initialized. Flag de indexación de directorio reseteado.")
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}", exc_info=True)

# ... (el bloque if __name__ == '__main__' necesita ser actualizado si se quiere probar esto directamente)
# ... (principalmente, el llamado a analyze_economic_impact ya no tomaría document_paths de la misma forma)