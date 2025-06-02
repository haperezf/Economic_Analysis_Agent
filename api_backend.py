# api_backend.py

import os
import sys
import logging
import re
from typing import List, Optional, Any, Dict, TypedDict # <--- Añadido TypedDict aquí
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Configuración de Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Cargar variables de entorno ---
load_dotenv()
logger.info("Variables de entorno cargadas desde .env (si existe).")

# --- Añadir la ruta del proyecto a sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    logger.info(f"Añadido al sys.path: {PROJECT_ROOT}")

# --- Importaciones de Langchain y EARIAAgent ---
try:
    from langchain_ollama import ChatOllama
except ImportError:
    logger.warning("langchain_ollama no encontrado, usando fallback. Considera 'pip install langchain-ollama'")
    from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

try:
    from earia_agent.agent_core.earia_agent import EARIAAgent, ANALYSIS_TYPE_COT, ANALYSIS_TYPE_BASIC_RAG, ANALYSIS_TYPE_TOT, ANALYSIS_TYPE_AIN_EXTRACT # Importa todos los tipos
except ImportError as e:
    logger.critical(f"Error al importar EARIAAgent: {e}. Revisa la ruta y PYTHONPATH.")
    exit()

# --- 1. Definición del Estado del Grafo (igual que en run_langgraph_earia_real.py) ---
class AgentWorkflowState(TypedDict): # Modificada
    original_query: str
    # document_paths: Optional[List[str]] # Ya no es input principal para el flujo general
    aspect_to_analyze: Optional[str]
    k_retrieval_for_earia: Optional[int]
    force_reprocess_docs_for_earia: Optional[bool] # Para el directorio completo
    analysis_type_for_earia: str
    earia_cot_result: Optional[str]
    interpretation_result: Optional[str]
    error_message: Optional[str]
    execution_log: List[str]

# --- 2. Configuración de los LLMs de Ollama (Igual que en run_langgraph_earia_real.py) ---
LLM_AGENT_2_INTERPRETER_MODEL = "gemma3" # Hardcodeado como solicitaste
OLLAMA_BASE_URL_AGENT2 = "http://localhost:11434"  # Hardcodeado

llm_interpreter_agent: Optional[ChatOllama] = None # Inicializar globalmente
earia_agent_instance: Optional[EARIAAgent] = None # Para la instancia de EARIAAgent

# --- Utilidad para limpiar salida del LLM (Igual) ---
def limpiar_bloques_think(texto: str) -> str:
    if not texto: return ""
    return re.sub(r"<think>.*?</think>\s*", "", texto, flags=re.DOTALL | re.IGNORECASE).strip()

# --- 3. Definición de Nodos del Grafo (Ligeramente adaptados) ---
# En api_backend.py (o run_langgraph_earia_real.py)

async def run_earia_agent_node(state: AgentWorkflowState) -> Dict[str, Any]:
    logger.info("--- Nodo: Ejecutando EARIA Agent (Análisis CoT) ---")
    current_log = state.get("execution_log", [])
    current_log.append("Iniciando análisis CoT con EARIAAgent...")
    try:
        query = state["original_query"]
        # 'document_paths' en el estado es para cualquier documento *adicional* específico.
        # Para el flujo donde se procesa el directorio por defecto, este será None.
        specific_docs_to_ensure = state.get("document_paths") 
        aspect = state.get("aspect_to_analyze")
        k_retrieval = state.get("k_retrieval_for_earia", 5) 
        # 'force_reprocess_docs_for_earia' en el estado ahora controla 
        # si se reprocesa todo el directorio por defecto.
        force_reprocess_main_directory = state.get("force_reprocess_docs_for_earia", False) 
        analysis_type = state.get("analysis_type_for_earia", ANALYSIS_TYPE_BASIC_RAG)

        logger.info(f"EARIAAgent - Query: '{query}', Tipo: {analysis_type}, Aspecto: '{aspect}', Docs Específicos: {specific_docs_to_ensure}, K: {k_retrieval}, ForzarReprocesoDir: {force_reprocess_main_directory}")

        global earia_agent_instance
        if earia_agent_instance is None:
            logger.error("Instancia de EARIAAgent no inicializada.")
            raise RuntimeError("EARIAAgent no inicializado.")
        
        result_text_raw = earia_agent_instance.analyze_economic_impact(
            query=query,
            document_paths_to_ensure=specific_docs_to_ensure, # <--- CORREGIDO AQUÍ
            analysis_type=analysis_type,
            aspect_to_analyze=aspect,
            k_retrieval=k_retrieval,
            force_reprocess_docs=force_reprocess_main_directory # Este se pasa para el directorio
        )

        # ... (resto de la lógica de manejo de errores y limpieza de <think> que ya tenías)
        if isinstance(result_text_raw, str) and \
           ("Error:" in result_text_raw and ("Failed to process" in result_text_raw or "Could not get a response" in result_text_raw or "document path does not exist" in result_text_raw or "Falló el procesamiento" in result_text_raw)):
            logger.error(f"EARIA Agent reportó un error: {result_text_raw}")
            current_log.append(f"Error reportado por EARIA Agent: {result_text_raw}")
            return {"error_message": result_text_raw, "earia_cot_result": None, "execution_log": current_log}

        cot_result_text_limpio = limpiar_bloques_think(result_text_raw)
        
        current_log.append(f"Análisis ({analysis_type}) generado por EARIAAgent (limpio): {cot_result_text_limpio[:300]}...")
        return {"earia_cot_result": cot_result_text_limpio, "execution_log": current_log, "error_message": None}

    except Exception as e:
        error_msg = f"Excepción crítica en run_earia_agent_node: {type(e).__name__} - {e}"
        logger.error(error_msg, exc_info=True)
        current_log.append(f"Error crítico en nodo EARIA: {error_msg}")
        return {"error_message": error_msg, "earia_cot_result": None, "execution_log": current_log}

async def interpret_cot_result_node(state: AgentWorkflowState) -> Dict[str, Any]:
    logger.info("--- Nodo: Interpretando Resultado (Agente 2 - Intérprete) ---")
    current_log = state.get("execution_log", [])
    current_log.append("Iniciando interpretación del resultado del Agente 1...")
    try:
        agent1_result = state.get("earia_cot_result")
        original_query = state.get("original_query")

        if state.get("error_message") and not agent1_result:
             logger.warning("Saltando interpretación debido a error en el nodo EARIA y no hay resultado.")
             current_log.append("Saltada interpretación por error previo y falta de resultado.")
             return {"execution_log": current_log} 

        if not agent1_result:
            error_msg = "No se encontró resultado del Agente 1 para interpretar."
            logger.error(error_msg)
            current_log.append(f"Error: {error_msg}")
            return {"error_message": error_msg, "interpretation_result": None, "execution_log": current_log}

        prompt_interpreter_str = (
            "Eres un economista senior con una excepcional capacidad para profundizar y expandir análisis económicos existentes, "
            "con un enfoque particular en el sector de telecomunicaciones si el contexto lo sugiere. Has recibido un informe de un analista (Agente 1). "
            "Este informe contiene un título general identificado, un resumen inicial del contexto documental, y un análisis económico detallado sobre la consulta original: '{original_query}'.\n\n"
            "--- INICIO INFORME DEL AGENTE 1 ---\n"
            "{texto_agente_1}\n" # texto_agente_1 contendrá: Título, Resumen Inicial, y el Análisis Detallado del Agente 1
            "--- FIN INFORME DEL AGENTE 1 ---\n\n"
            "Tu tarea es tomar la sección de 'Análisis Detallado' proporcionada por el Agente 1 y realizar una EXPANSIÓN ANALÍTICA PROFUNDA Y PROFESIONAL en español. Tu contribución debe añadir valor significativo, llevando el análisis inicial a un nivel superior de detalle y perspicacia económica. Enfócate en:\n"
            "1.  **Identificación y Elaboración de Implicaciones Estratégicas No Obvias:** A partir de los hallazgos del Agente 1, extrae y detalla exhaustivamente 2-3 implicaciones estratégicas (económicas, de mercado, de política, para diferentes stakeholders en telecomunicaciones, etc.) que consideres más críticas o de mayor alcance a largo plazo, y que quizás no fueron completamente desarrolladas por el Agente 1. Justifica tu selección y profundiza en sus posibles consecuencias.\n"
            "2.  **Desarrollo de Escenarios Futuros Detallados y Justificados:** Basado en el análisis del Agente 1, y considerando los factores económicos clave, desarrolla 1-2 escenarios futuros plausibles (ej. optimista, pesimista, o un escenario de disrupción tecnológica/regulatoria) con un nivel de detalle considerable. Describe los supuestos de cada escenario y sus posibles consecuencias económicas específicas para el sector de telecomunicaciones y actores relacionados.\n"
            "3.  **Análisis de Interconexiones y Efectos Sistémicos:** Profundiza en las interconexiones entre los diferentes factores económicos y los impactos mencionados por el Agente 1. Explora y detalla posibles efectos de segundo o tercer orden, o efectos sistémicos en la economía o el sector de telecomunicaciones, que podrían no haber sido el foco principal del Agente 1 pero que son relevantes.\n"
            "4.  **Propuesta de Métricas o Indicadores Clave de Seguimiento (KPIs):** Si el análisis se refiere a políticas, proyectos o tendencias, ¿qué métricas o indicadores económicos clave se deberían monitorear para evaluar la evolución real de los impactos discutidos? Justifica tu propuesta.\n"
            "5.  **Síntesis de Profundización y Valor Agregado:** Concluye con una síntesis clara de los nuevos insights, la profundidad adicional, o las perspectivas estratégicas que tu análisis aporta sobre el trabajo del Agente 1.\n\n"
            "Tu respuesta debe ser extensa, detallada, analítica y reflejar una perspectiva económica experta y profunda. No te limites a parafrasear al Agente 1; construye sobre su trabajo y llévalo a un nivel superior. Estructura tu respuesta claramente, usando los puntos numerados como guía. "
            "Evita meta-comentarios sobre tu propio proceso de pensamiento en la respuesta final."
        )
        prompt_interpreter = ChatPromptTemplate.from_template(prompt_interpreter_str)
        
        global llm_interpreter_agent # Usar instancia global
        if llm_interpreter_agent is None:
            raise RuntimeError("LLM Intérprete no inicializado.")

        chain_interpreter = prompt_interpreter | llm_interpreter_agent | StrOutputParser()
        
        logger.info(f"Agente Intérprete - Realizando meta-análisis (input: {len(agent1_result)} chars)...")
        
        # Usar ainvoke para la cadena del intérprete
        interpretation_raw = await chain_interpreter.ainvoke({
            "texto_agente_1": agent1_result,
            "original_query": original_query
        })
        interpretation_limpia = limpiar_bloques_think(interpretation_raw)

        current_log.append(f"Meta-análisis generado (limpio): {interpretation_limpia[:300]}...")
        return {"interpretation_result": interpretation_limpia, "execution_log": current_log, "error_message": None}

    except Exception as e:
        error_msg = f"Excepción crítica en interpret_cot_result_node: {type(e).__name__} - {e}"
        logger.error(error_msg, exc_info=True)
        current_log.append(f"Error crítico en nodo Intérprete: {error_msg}")
        return {"error_message": error_msg, "interpretation_result": None, "execution_log": current_log}

# --- 4. Construcción del Grafo (Global para que se compile una vez) ---
workflow_builder = StateGraph(AgentWorkflowState)
workflow_builder.add_node("earia_analysis_step", run_earia_agent_node)
workflow_builder.add_node("interpretation_step", interpret_cot_result_node)
workflow_builder.set_entry_point("earia_analysis_step")
workflow_builder.add_edge("earia_analysis_step", "interpretation_step")
workflow_builder.add_edge("interpretation_step", END)

app_langgraph = workflow_builder.compile()
logger.info("Grafo de LangGraph compilado.")

# --- 5. Definición de la API con FastAPI ---
app_fastapi = FastAPI(title="EARIA Agent API")

# Configuración de CORS
app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite todos los orígenes (para desarrollo). En producción, sé más restrictivo.
    allow_credentials=True,
    allow_methods=["*"], # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"], # Permite todos los headers
)

class AnalysisRequest(BaseModel): # Modificada
    query: str
    # document_paths: Optional[List[str]] = None # Eliminado
    aspect_to_analyze: Optional[str] = None
    analysis_type: str = ANALYSIS_TYPE_BASIC_RAG
    k_retrieval: int = 7 # Default aumentado
    force_reprocess_entire_directory: bool = False # Nuevo campo

class AnalysisResponse(BaseModel):
    original_query: str
    earia_result: Optional[str]
    interpreter_result: Optional[str]
    execution_log: Optional[List[str]]
    error_message: Optional[str]


@app_fastapi.on_event("startup")
async def startup_event():
    global llm_interpreter_agent, earia_agent_instance
    logger.info("Evento startup de FastAPI: Inicializando LLM Intérprete y EARIAAgent...")
    try:
        llm_interpreter_agent = ChatOllama(
            model=LLM_AGENT_2_INTERPRETER_MODEL,
            base_url=OLLAMA_BASE_URL_AGENT2,
            temperature=0.2
        )
        logger.info(f"LLM Intérprete ({LLM_AGENT_2_INTERPRETER_MODEL}) inicializado.")
        
        earia_agent_instance = EARIAAgent() # EARIAAgent usa su propia config de LLM
        logger.info("Instancia de EARIAAgent creada.")

    except Exception as e:
        logger.critical(f"Error fatal durante el startup de FastAPI al inicializar componentes: {e}", exc_info=True)
        # Podrías querer que la aplicación falle si los componentes críticos no se pueden cargar.
        # raise # Descomenta para hacer que la app falle si esto ocurre.


@app_fastapi.post("/analyze", response_model=AnalysisResponse)
async def analyze_documents_endpoint(request: AnalysisRequest):
    logger.info(f"Recibida petición /analyze: {request.model_dump(exclude_none=True)}")
    # ... (check de inicialización de agentes) ...

    initial_state = AgentWorkflowState(
        original_query=request.query,
        document_paths=None, # EARIAAgent usará su directorio configurado
        aspect_to_analyze=request.aspect_to_analyze,
        k_retrieval_for_earia=request.k_retrieval,
        force_reprocess_docs_for_earia=request.force_reprocess_entire_directory, # Mapeado
        analysis_type_for_earia=request.analysis_type,
        earia_cot_result=None,
        interpretation_result=None,
        error_message=None,
        execution_log=[]
    )

    try:
        # Usar ainvoke para la ejecución asíncrona del grafo
        final_configs = {"recursion_limit": 15}
        final_state = await app_langgraph.ainvoke(initial_state, config=final_configs)
        
        logger.info("Procesamiento del grafo completado.")
        # logger.debug(f"Estado final del grafo: {final_state}") # Puede ser muy verboso

        if not isinstance(final_state, dict): # Chequeo por si ainvoke devuelve algo inesperado
            logger.error(f"El estado final del grafo no es un diccionario: {final_state}")
            raise HTTPException(status_code=500, detail="Error interno: El grafo no devolvió un estado válido.")


        return AnalysisResponse(
            original_query=final_state.get("original_query", request.query),
            earia_result=final_state.get("earia_cot_result"),
            interpreter_result=final_state.get("interpretation_result"),
            execution_log=final_state.get("execution_log"),
            error_message=final_state.get("error_message")
        )
    except Exception as e:
        logger.error(f"Excepción durante la invocación del grafo LangGraph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error procesando la solicitud: {e}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando servidor FastAPI con Uvicorn...")
    # Para desarrollo, uvicorn.run() es conveniente. En producción, usarías un comando como:
    # uvicorn api_backend:app_fastapi --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000, log_level="info")