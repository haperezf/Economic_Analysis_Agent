# earia_agent/agent_core/llm_handler.py

import os
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama # Fallback

# Importar los prompts (los nombres de las variables de prompt deben coincidir)
from ..prompts.economic_impact_prompts import (
    BASIC_RAG_CHAT_PROMPT,
    get_cot_analysis_prompt, # Nombre de función consistente con el archivo de prompts
    TOT_EXPLORATION_CHAT_PROMPT,
    EXTRACT_AIN_COMPONENTS_CHAT_PROMPT
)
import logging

module_logger = logging.getLogger(__name__ + ".config")
logger = logging.getLogger(__name__)

# --- Configuración de Variables de Entorno ---
raw_env_ollama_model = os.getenv("OLLAMA_MODEL")
cleaned_ollama_model = ""
if raw_env_ollama_model:
    cleaned_ollama_model = raw_env_ollama_model.split('#')[0].strip().strip('"').strip("'")
module_logger.info(f"Raw OLLAMA_MODEL (limpio) from environment: '{cleaned_ollama_model}'")

#DEFAULT_OLLAMA_MODEL = cleaned_ollama_model if cleaned_ollama_model else "gemma3"
DEFAULT_OLLAMA_MODEL = "gemma3"
module_logger.info(f"Effective DEFAULT_OLLAMA_MODEL for this LLMHandler: '{DEFAULT_OLLAMA_MODEL}'")

raw_env_ollama_base_url = os.getenv("OLLAMA_BASE_URL")
cleaned_ollama_base_url = ""
if raw_env_ollama_base_url:
    cleaned_ollama_base_url = raw_env_ollama_base_url.split('#')[0].strip().strip('"').strip("'")
module_logger.info(f"Raw OLLAMA_BASE_URL (limpio) from environment: '{cleaned_ollama_base_url}'")

DEFAULT_OLLAMA_BASE_URL = cleaned_ollama_base_url if cleaned_ollama_base_url else "http://localhost:11434"
module_logger.info(f"Effective DEFAULT_OLLAMA_BASE_URL for this LLMHandler: '{DEFAULT_OLLAMA_BASE_URL}'")


class LLMHandler:
    def __init__(self,
                 model_name: str = DEFAULT_OLLAMA_MODEL,
                 base_url: str = DEFAULT_OLLAMA_BASE_URL,
                 temperature: float = 0.2, 
                 num_ctx: int = 16384): # Aumentado por defecto para análisis profundos
        
        logger.info(f"LLMHandler __init__ called with model_name: '{model_name}', base_url: '{base_url}', temp: {temperature}, num_ctx: {num_ctx}")
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx

        if not self.model_name:
            error_msg = "LLMHandler error: model_name está vacío o es None. No se puede inicializar ChatOllama."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Attempting to initialize ChatOllama with model: '{self.model_name}' at base_url: '{self.base_url}'")
        try:
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                num_ctx=self.num_ctx,
                request_timeout=300.0 # Añadido timeout más largo para respuestas verbosas
            )
            logger.info(f"ChatOllama initialized successfully with model: {self.model_name}.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOllama with model '{self.model_name}': {e}", exc_info=True)
            raise
        
        self.output_parser = StrOutputParser()

    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        if not context_docs:
            return "No se encontró contexto documental relevante." # Mensaje más general
        
        formatted_context_parts = []
        for i, doc_chunk in enumerate(context_docs): # Renombrado doc a doc_chunk para claridad
            page_content = doc_chunk.get("page_content", "Contenido Faltante")
            metadata = doc_chunk.get("metadata", {})
            source = metadata.get('source', 'Fuente Desconocida')
            # 'original_page' y 'chunk_id' son útiles si provienen del text_processor
            page_info = f"Página: {metadata.get('original_document_page_index', 'N/A')}" if 'original_document_page_index' in metadata else ""
            chunk_id_info = f"Chunk ID: {metadata.get('chunk_id', f'extracto_{i}')}"
            
            header = f"--- Extracto de Documento {i+1} (Fuente: {source}, {page_info} {chunk_id_info}) ---"
            formatted_context_parts.append(f"{header}\n{page_content}\n")
            
        return "\n".join(formatted_context_parts)

    def generate_title_and_summary(self, combined_context_text: str, original_query: str) -> str:
        if not combined_context_text or not combined_context_text.strip():
            return ("Título General Identificado: No disponible (contexto documental insuficiente o vacío)\n\n"
                    "Resumen Inicial del Contexto Documental:\nNo se proporcionó contexto documental válido para resumir.")

        logger.debug(f"Generando título y resumen inicial para la consulta: '{original_query}' y contexto de {len(combined_context_text)} caracteres.")
        
        prompt_str = (
            "Dada la consulta del usuario: '{original_query}', y el siguiente texto compilado de varios documentos o secciones de documentos relevantes:\n"
            "--- INICIO DEL TEXTO DE CONTEXTO DOCUMENTAL ---\n"
            "{context_text}\n"
            "--- FIN DEL TEXTO DE CONTEXTO DOCUMENTAL ---\n\n"
            "Por favor, realiza las siguientes dos tareas en español, con un enfoque analítico y verboso:\n"
            "1. Identifica o infiere un TÍTULO GENERAL descriptivo y completo que englobe el tema central o los temas principales cubiertos por el texto de contexto documental, especialmente en relación con la consulta del usuario.\n"
            "2. Genera un RESUMEN INICIAL SUSTANCIAL Y DETALLADO (aproximadamente 5-7 frases bien elaboradas o dos párrafos cortos) de los puntos, argumentos o datos clave más importantes presentes en el texto de contexto documental que sean pertinentes para la consulta del usuario. Este resumen debe ofrecer una visión general sólida del contenido que se analizará más a fondo.\n\n"
            "Formatea tu respuesta estrictamente así, sin explicaciones adicionales sobre cómo lo hiciste:\n"
            "Título General Identificado: [Tu título aquí]\n\n"
            "Resumen Inicial del Contexto Documental:\n[Tu resumen aquí]"
        )
        
        title_summary_prompt = ChatPromptTemplate.from_template(prompt_str)
        chain = title_summary_prompt | self.llm | self.output_parser 

        try:
            response = chain.invoke({
                "original_query": original_query,
                "context_text": combined_context_text
            })
            logger.debug(f"LLM (título y resumen) respuesta (primeros 300 chars): {response[:300]}...")
            return response
        except Exception as e:
            logger.error(f"Error durante la generación de título y resumen inicial: {e}", exc_info=True)
            return ("Título General Identificado: Error en la generación\n\n"
                    "Resumen Inicial del Contexto Documental:\nError al generar el resumen debido a una excepción.")

    def generate_response_rag(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        logger.debug(f"Generando respuesta RAG profunda para query: '{query}'")
        formatted_context = self._format_context(context_docs)
        
        prompt = BASIC_RAG_CHAT_PROMPT # Usa el prompt generalizado y de análisis profundo
        if not isinstance(prompt, ChatPromptTemplate):
            logger.error("BASIC_RAG_CHAT_PROMPT no es una instancia válida de ChatPromptTemplate.")
            return "Error: Problema de configuración del prompt RAG."
            
        chain = prompt | self.llm | self.output_parser
        
        try:
            response = chain.invoke({"query": query, "context": formatted_context})
            logger.debug(f"LLM RAG response (primeros 300 chars): {str(response)[:300]}...")
            return str(response)
        except Exception as e:
            logger.error(f"Error durante la llamada LLM para RAG: {e}", exc_info=True)
            return "Error: No se pudo obtener una respuesta del LLM para RAG."

    def generate_response_cot_analysis(self, query: str, context_docs: List[Dict[str, Any]], aspect_to_analyze: str) -> str:
        logger.debug(f"Generando análisis CoT profundo para query: '{query}', aspecto/tema principal: '{aspect_to_analyze}'")
        formatted_context = self._format_context(context_docs)
        
        try:
            # Usa el prompt CoT generalizado y de análisis profundo
            cot_prompt = get_cot_analysis_prompt(aspect_to_analyze=aspect_to_analyze) 
        except Exception as e:
            logger.error(f"Error obteniendo el prompt CoT: {e}", exc_info=True)
            return "Error: No se pudo configurar el prompt CoT."

        if not isinstance(cot_prompt, ChatPromptTemplate):
            logger.error("get_cot_analysis_prompt no devolvió un ChatPromptTemplate válido.")
            return "Error: Problema de configuración del prompt CoT."

        chain = cot_prompt | self.llm | self.output_parser
        
        try:
            response = chain.invoke({"query": query, "context": formatted_context})
            logger.debug(f"LLM CoT response (primeros 300 chars): {str(response)[:300]}...")
            return str(response)
        except Exception as e:
            logger.error(f"Error durante la llamada LLM para CoT: {e}", exc_info=True)
            return "Error: No se pudo obtener una respuesta CoT del LLM."

    def generate_response_tot_exploration(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        logger.debug(f"Generando exploración ToT profunda para query: '{query}'")
        formatted_context = self._format_context(context_docs)
        
        prompt = TOT_EXPLORATION_CHAT_PROMPT # Usa el prompt ToT generalizado y de análisis profundo
        if not isinstance(prompt, ChatPromptTemplate):
            logger.error("TOT_EXPLORATION_CHAT_PROMPT no es una instancia válida de ChatPromptTemplate.")
            return "Error: Problema de configuración del prompt ToT."
            
        chain = prompt | self.llm | self.output_parser
        
        try:
            response = chain.invoke({"query": query, "context": formatted_context})
            logger.debug(f"LLM ToT response (primeros 300 chars): {str(response)[:300]}...")
            return str(response)
        except Exception as e:
            logger.error(f"Error durante la llamada LLM para ToT: {e}", exc_info=True)
            return "Error: No se pudo obtener una respuesta ToT del LLM."

    def generate_response_extract_ain(self, context_docs: List[Dict[str, Any]], query_for_ain: Optional[str] = None) -> str:
        logger.debug(f"Generando extracción AIN detallada. Query provista para AIN: '{query_for_ain if query_for_ain else 'Ninguna'}'")
        formatted_context = self._format_context(context_docs)
        
        prompt = EXTRACT_AIN_COMPONENTS_CHAT_PROMPT # Usa el prompt AIN más verboso
        if not isinstance(prompt, ChatPromptTemplate):
            logger.error("EXTRACT_AIN_COMPONENTS_CHAT_PROMPT no es una instancia válida de ChatPromptTemplate.")
            return "Error: Problema de configuración del prompt AIN."

        chain = prompt | self.llm | self.output_parser
        
        try:
            input_data = {"context": formatted_context}
            response = chain.invoke(input_data)
            logger.debug(f"LLM AIN extraction response (primeros 300 chars): {str(response)[:300]}...")
            return str(response)
        except Exception as e:
            logger.error(f"Error durante la llamada LLM para extracción AIN: {e}", exc_info=True)
            return "Error: No se pudo obtener una respuesta de extracción AIN del LLM."