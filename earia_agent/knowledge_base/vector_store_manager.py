# earia_agent/knowledge_base/vector_store_manager.py

import os
import logging
import re # No se usa directamente aquí, pero TextProcessor sí lo usa.
import unicodedata # Para limpieza avanzada de Unicode
from typing import List, Optional, Any, Dict

# Definir el logger INMEDIATAMENTE después de importar logging
logger = logging.getLogger(__name__)
# La configuración básica de logging (basicConfig) se hace en el punto de entrada.

# Nuevas importaciones recomendadas por LangChain
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    logger.warning("langchain-huggingface no encontrado. Usando fallback. Considera: pip install langchain-huggingface")
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    logger.warning("langchain-chroma no encontrado. Usando fallback. Considera: pip install langchain-chroma")
    from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever 

import chromadb 
import shutil 

# --- Configuración por Defecto ---
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./earia_db/chroma_store_vsm_v2") # Nueva ruta para prueba limpia
DEFAULT_COLLECTION_NAME = os.getenv("DEFAULT_CHROMA_COLLECTION", "earia_docs_vsm_v2")


def clean_unicode_text(text: str) -> str:
    """
    Normaliza Unicode y elimina la mayoría de los caracteres de control,
    preservando espacios en blanco comunes.
    """
    if not isinstance(text, str):
        # Esto no debería suceder si la sanitización previa funciona, pero por si acaso.
        logger.warning(f"clean_unicode_text recibió tipo no-string: {type(text)}. Devolviendo cadena vacía.")
        return ""
    
    # NFKC es una forma de normalización que descompone caracteres de compatibilidad
    # y luego los recompone canónicamente. Es bueno para consistencia.
    normalized_text = unicodedata.normalize('NFKC', text)
    
    cleaned_chars = []
    for char in normalized_text:
        category = unicodedata.category(char)
        # Cc: Other, Control
        # Cf: Other, Format (ej. zero-width non-joiner)
        # Co: Other, Private Use
        # Cn: Other, Not Assigned (Unassigned)
        # Zl: Separator, Line
        # Zp: Separator, Paragraph
        # Estos últimos dos (Zl, Zp) a veces son problemáticos si no se convierten a \n.
        # \s+ en TextProcessor.clean_text debería manejarlos.
        if category.startswith('C') or category in ('Zl', 'Zp'): # Eliminar controles, formatos, privados, no asignados, y separadores de línea/párrafo explícitos
            if char not in ('\n', '\r', '\t'): # Mantener saltos de línea, retornos de carro, tabs
                cleaned_chars.append(' ') # Reemplazar otros con un espacio
            else:
                cleaned_chars.append(char) # Mantener \n, \r, \t
        else:
            cleaned_chars.append(char) # Mantener todos los demás caracteres (letras, números, puntuación, símbolos, etc.)
    
    return "".join(cleaned_chars)


class VectorStoreManager:
    def __init__(self,
                 embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
                 persist_directory: str = DEFAULT_VECTOR_DB_PATH,
                 collection_name: str = DEFAULT_COLLECTION_NAME):
        logger.info(f"Inicializando VectorStoreManager con modelo de embedding: {embedding_model_name}")
        logger.info(f"Directorio de persistencia de ChromaDB: {os.path.abspath(persist_directory)}")
        logger.info(f"Nombre de la colección en ChromaDB: {collection_name}")

        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
        except OSError as e:
            logger.error(f"Fallo al crear el directorio de persistencia {self.persist_directory}: {e}", exc_info=True)
            raise

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'}, 
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Modelo de embeddings '{self.embedding_model_name}' cargado exitosamente.")
        except Exception as e:
            logger.error(f"Fallo al cargar el modelo de embeddings HuggingFace '{self.embedding_model_name}': {e}", exc_info=True)
            raise
        
        try:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"}) 
            
            self.vector_store = Chroma(
                client=self._client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            logger.info(f"VectorStoreManager inicializado. Conteo inicial de la colección '{self.collection_name}': {self.get_collection_count()}")
        except Exception as e:
            logger.error(f"Fallo al inicializar el cliente Chroma o el vector store: {e}", exc_info=True)
            raise

    def get_collection_count(self) -> int:
        try:
            collection = self._client.get_collection(name=self.collection_name) # Lanza error si no existe en versiones recientes
            return collection.count()
        except Exception: # Captura más genérica si get_collection falla por no existencia
            logger.warning(f"Colección '{self.collection_name}' no encontrada o error al acceder. Asumiendo 0.")
            # Intentar crearla aquí si es la primera vez y no existe, puede ser una opción.
            try:
                self._client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
                return 0 # Recién creada, así que 0
            except Exception as e_create:
                logger.error(f"Fallo al intentar crear la colección '{self.collection_name}' después de no encontrarla: {e_create}")
                return 0 # O -1 para indicar un problema más serio

    def add_documents(self, documents: List[Document], batch_size: Optional[int] = None) -> bool:
        if not documents:
            logger.warning("No se proporcionaron documentos para añadir al vector store.")
            return True 

        logger.info(f"Recibidos {len(documents)} documentos para posible adición a la colección '{self.collection_name}'.")
        
        texts_for_embedding: List[str] = []
        metadatas_for_embedding: List[Dict[str, Any]] = []
        ids_for_embedding: List[str] = []
        
        problematic_sources_to_debug = [
            "documento-respuesta-a-comentarios", # Parte del nombre de tus archivos problemáticos
            "documento-soporte-medidas-moviles"  # Parte del nombre de tus archivos problemáticos
        ]

        for i, doc_input in enumerate(documents): # doc_input es el Document original del chunker
            page_content_original = doc_input.page_content
            current_metadata = doc_input.metadata if hasattr(doc_input, 'metadata') and isinstance(doc_input.metadata, dict) else {}
            source_file = current_metadata.get('source', 'Desconocida').lower()
            
            # Generación de ID
            doc_id_str: str
            if current_metadata.get("chunk_id"):
                doc_id_str = str(current_metadata.get("chunk_id"))
            else:
                # Si page_content_original es None o no es string, fallback_content_for_hash será ""
                fallback_content_for_hash = page_content_original if isinstance(page_content_original, str) else ""
                doc_id_str = f"auto_gen_id_{i}_{hash(fallback_content_for_hash)}"
                current_metadata["generated_chunk_id"] = doc_id_str
            
            # Sanitización del page_content
            page_content_processed: str = "" # Default a cadena vacía
            if page_content_original is None:
                logger.debug(f"Chunk ID '{doc_id_str}' (Fuente: {source_file}) tiene page_content=None. Se procesará como cadena vacía.")
                # page_content_processed ya es ""
            elif isinstance(page_content_original, str):
                page_content_processed = page_content_original
            else: 
                logger.warning(f"Chunk ID '{doc_id_str}' (Fuente: {source_file}) tiene page_content de tipo {type(page_content_original)}. Intentando convertir a string.")
                try:
                    page_content_processed = str(page_content_original)
                except Exception as e_conv:
                    logger.error(f"No se pudo convertir page_content a string para el chunk ID '{doc_id_str}'. Saltando este chunk. Error: {e_conv}")
                    continue 

            # Limpieza adicional de Unicode y luego de espacios
            page_content_cleaned_unicode = clean_unicode_text(page_content_processed)
            page_content_final_for_embedding = page_content_cleaned_unicode.strip() # Quitar espacios al inicio/final

            # Logging de depuración para archivos problemáticos
            for problematic_keyword in problematic_sources_to_debug:
                if problematic_keyword in source_file:
                    logger.debug(f"DEBUG CHUNK (Fuente: {source_file}, ID: {doc_id_str}):\n"
                                 f"  Tipo Original: {type(page_content_original)}\n"
                                 f"  Tipo Procesado (antes de strip): {type(page_content_cleaned_unicode)}\n"
                                 f"  Tipo Final: {type(page_content_final_for_embedding)}\n"
                                 f"  Longitud Final: {len(page_content_final_for_embedding)}\n"
                                 f"  Contenido Final (primeros 200): '{page_content_final_for_embedding[:200]}'")
                    if not page_content_final_for_embedding:
                         logger.debug(f"    Este chunk de fuente problemática resultó vacío después de la limpieza y strip.")
                    break # Solo loguear una vez por chunk de fuente problemática

            if page_content_final_for_embedding: # Solo añadir si hay contenido textual real
                texts_for_embedding.append(page_content_final_for_embedding)
                metadatas_for_embedding.append(current_metadata) # Usar metadata que podría tener generated_chunk_id
                ids_for_embedding.append(doc_id_str)
            else:
                logger.info(f"Chunk ID '{doc_id_str}' (Fuente: {source_file}) omitido porque su contenido es vacío después de toda la limpieza.")

        if not texts_for_embedding:
            logger.warning("No hay documentos con contenido de texto válido para añadir después de la sanitización.")
            return True 

        logger.info(f"Añadiendo {len(texts_for_embedding)} chunks con contenido válido a la colección '{self.collection_name}'...")
        try:
            self.vector_store.add_texts( # Usamos add_texts ya que hemos preparado las listas
                texts=texts_for_embedding, 
                metadatas=metadatas_for_embedding, 
                ids=ids_for_embedding
            )
            logger.info(f"Se añadieron exitosamente {len(texts_for_embedding)} chunks. La colección ahora tiene {self.get_collection_count()} items.")
            return True
        except chromadb.errors.IDAlreadyExistsError as id_error:
            logger.error(f"Error: Uno o más IDs de documento ya existen. Detalles: {id_error}", exc_info=True)
            return False
        except TypeError as te: 
            logger.error(f"TypeError durante add_texts (probablemente en la función de embedding): {te}", exc_info=True)
            logger.error("Esto usualmente indica que algún texto en la lista 'texts_for_embedding' todavía no es un string válido para el tokenizador, a pesar de la sanitización.")
            # Loguear los tipos de los primeros elementos que causaron problemas podría ayudar
            for i_err, txt_err in enumerate(texts_for_embedding):
                if not isinstance(txt_err, str):
                    logger.error(f"  Elemento NO STRING encontrado en texts_for_embedding en el índice {i_err}, ID: {ids_for_embedding[i_err]}, Tipo: {type(txt_err)}, Valor: {str(txt_err)[:100]}")
                    # Romper después del primer error para no inundar logs si son muchos
                    break 
            return False
        except Exception as e:
            logger.error(f"Error añadiendo textos a ChromaDB: {e}", exc_info=True)
            return False

    def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        logger.debug(f"Realizando búsqueda por similitud para query: '{query}' con k={k}, filter={filter}")
        try:
            if self.vector_store is None:
                logger.error("Vector store no inicializado. No se puede realizar la búsqueda.")
                return []
            results = self.vector_store.similarity_search(query, k=k, filter=filter)
            logger.debug(f"Encontrados {len(results)} documentos similares para la query.")
            return results
        except Exception as e:
            logger.error(f"Error durante la búsqueda por similitud: {e}", exc_info=True)
            return []
            
    def get_retriever(self, 
                      search_type: str = "similarity", 
                      search_kwargs: Optional[Dict[str, Any]] = None
                     ) -> VectorStoreRetriever:
        if search_kwargs is None:
            search_kwargs = {'k': 5} 
        
        logger.info(f"Creando retriever con search_type='{search_type}' y search_kwargs={search_kwargs}")
        if self.vector_store is None:
            logger.error("Vector store no inicializado. No se puede crear el retriever.")
            raise RuntimeError("VectorStore no está disponible para crear un retriever.")
            
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def clear_collection(self) -> None:
        logger.warning(f"Intentando limpiar todos los documentos de la colección '{self.collection_name}' eliminando y recreando la colección.")
        try:
            self._client.delete_collection(name=self.collection_name)
            logger.info(f"Colección '{self.collection_name}' eliminada exitosamente.")
        except ValueError as ve: 
            if "does not exist" in str(ve).lower() or "no collection named" in str(ve).lower() or "could not find collection" in str(ve).lower() :
                logger.info(f"Colección '{self.collection_name}' no existía, nada que eliminar.")
            else: 
                logger.error(f"ValueError al intentar eliminar la colección '{self.collection_name}': {ve}", exc_info=True)
        except chromadb.errors.CollectionNotDefinedError:
             logger.info(f"Colección '{self.collection_name}' no definida, nada que eliminar.")
        except Exception as e:
            logger.error(f"Error inesperado al intentar eliminar la colección '{self.collection_name}': {e}", exc_info=True)

        try:
            self._client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
            self.vector_store = Chroma(
                client=self._client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            logger.info(f"Colección '{self.collection_name}' re-inicializada/obtenida. Conteo: {self.get_collection_count()}")
        except Exception as e:
            logger.error(f"Fallo al re-inicializar el vector store de Chroma después de limpiar la colección: {e}", exc_info=True)
            raise

# Ejemplo de uso directo (opcional)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')
    
    test_db_dir = "./test_chroma_db_vsm_main_file_v3"
    test_collection_name = "vsm_test_collection_main_file_v3"

    if os.path.exists(test_db_dir):
        logger.info(f"Eliminando directorio de BD de prueba existente: {test_db_dir}")
        shutil.rmtree(test_db_dir)

    logger.info("--- Probando VectorStoreManager ---")
    try:
        manager = VectorStoreManager(persist_directory=test_db_dir, collection_name=test_collection_name)
        logger.info(f"Conteo inicial: {manager.get_collection_count()}")

        docs_to_add = [
            Document(page_content="Texto normal y corriente.", metadata={"source": "test_normal.txt", "chunk_id": "normal_0"}),
            Document(page_content=None, metadata={"source": "test_none.txt", "chunk_id": "none_0"}),
            Document(page_content=12345, metadata={"source": "test_int.txt", "chunk_id": "int_0"}),
            Document(page_content="   \t\n   ", metadata={"source": "test_whitespace.txt", "chunk_id": "whitespace_0"}),
            Document(page_content="Texto con \u0000 null char y \u001a EOF.", metadata={"source": "test_control_chars.txt", "chunk_id": "control_0"})
        ]
        add_success = manager.add_documents(docs_to_add)
        logger.info(f"Resultado de add_documents: {add_success}")
        current_count = manager.get_collection_count()
        logger.info(f"Conteo después de añadir: {current_count}") 

        if current_count > 0:
            results = manager.similarity_search("texto", k=3)
            logger.info("\nResultados de búsqueda por 'texto':")
            if results:
                for res_doc in results:
                    logger.info(f"- Contenido: '{res_doc.page_content[:50]}...', Metadata: {res_doc.metadata}")
            else:
                logger.info("  No se encontraron resultados.")
        else:
            logger.warning("No hay documentos en la colección para probar la búsqueda.")
            
        # Probar limpiar colección
        logger.info("\nLimpiando colección...")
        manager.clear_collection()
        logger.info(f"Conteo después de limpiar: {manager.get_collection_count()}")

    except Exception as e:
        logger.error(f"Error durante las pruebas de VectorStoreManager: {e}", exc_info=True)
    finally:
        if os.path.exists(test_db_dir):
            logger.info(f"Eliminando directorio de BD de prueba: {test_db_dir}")
            # shutil.rmtree(test_db_dir) # Comentado para poder inspeccionar la BD si es necesario
            logger.info(f"Directorio de prueba '{test_db_dir}' no eliminado para inspección.")
    logger.info("--- Fin de pruebas de VectorStoreManager ---")