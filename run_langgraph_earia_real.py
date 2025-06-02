# run_langgraph_earia_real.py

import os
import sys
import logging
import re # Para limpiar los bloques <think>
from typing import TypedDict, List, Optional, Any, Dict
from dotenv import load_dotenv

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

# --- Añadir la ruta del proyecto a sys.path si es necesario ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    logger.info(f"Añadido al sys.path: {PROJECT_ROOT}")
else:
    logger.info(f"PROJECT_ROOT ({PROJECT_ROOT}) ya está en sys.path.")

# --- Importaciones de Langchain y EARIAAgent ---
# Recordatorio: LangChain está moviendo componentes a paquetes específicos.
# Considera actualizar tus importaciones y dependencias en EARIAAgent también:
# pip install -U langchain-ollama langchain-huggingface langchain-chroma
try:
    from langchain_ollama import ChatOllama
except ImportError:
    logger.warning("langchain_ollama no encontrado, intentando con langchain_community.chat_models.ChatOllama.")
    logger.warning("Considera ejecutar: pip install langchain-ollama")
    from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

try:
    from earia_agent.agent_core.earia_agent import EARIAAgent, ANALYSIS_TYPE_COT
except ImportError as e:
    logger.critical(f"Error al importar EARIAAgent: {e}. Asegúrate de que el script se ejecute desde el directorio raíz del proyecto "
                  f"o que 'earia_agent' esté correctamente en PYTHONPATH.")
    exit()

# --- 1. Definición del Estado del Grafo ---
class AgentWorkflowState(TypedDict):
    original_query: str
    document_paths: Optional[List[str]]
    aspect_to_analyze: str
    k_retrieval_for_earia: Optional[int]
    force_reprocess_docs_for_earia: Optional[bool]
    earia_cot_result: Optional[str]
    interpretation_result: Optional[str]
    error_message: Optional[str]
    execution_log: List[str]

# --- 2. Configuración de los LLMs de Ollama ---
# El LLM para EARIAAgent se configura internamente (leyendo OLLAMA_MODEL de .env).
# LLM para el Agente 2 (Intérprete) con valores FIJOS (hardcodeados).
LLM_AGENT_2_INTERPRETER_MODEL = "tripolskypetr/gemma3-tools:4b"
OLLAMA_BASE_URL_AGENT2 = "http://localhost:11434"

logger.info(f"Modelo para Agente 2 (Intérprete) FIJADO A: {LLM_AGENT_2_INTERPRETER_MODEL} en {OLLAMA_BASE_URL_AGENT2}")

try:
    llm_interpreter_agent = ChatOllama(
        model=LLM_AGENT_2_INTERPRETER_MODEL,
        base_url=OLLAMA_BASE_URL_AGENT2,
        temperature=0.3,
        num_ctx=8192, # O el valor apropiado para phi4-mini
        request_timeout=300.0 
    )
except Exception as e:
    logger.critical(f"Error inicializando modelo Ollama para Agente Intérprete ({LLM_AGENT_2_INTERPRETER_MODEL}): {e}", exc_info=True)
    logger.critical(f"Asegúrate de que Ollama esté corriendo y el modelo especificado ('{LLM_AGENT_2_INTERPRETER_MODEL}') esté descargado.")
    logger.critical("Puedes descargar modelos con 'ollama pull nombre_del_modelo'.")
    exit()

# --- Utilidad para limpiar salida del LLM ---
def limpiar_bloques_think(texto: str) -> str:
    if not texto:
        return ""
    texto_limpio = re.sub(r"<think>.*?</think>\s*", "", texto, flags=re.DOTALL | re.IGNORECASE)
    return texto_limpio.strip()

# --- 3. Definición de Nodos del Grafo ---
def run_earia_agent_node(state: AgentWorkflowState) -> Dict[str, Any]:
    logger.info("--- Nodo: Ejecutando EARIA Agent (Análisis CoT) ---")
    current_log = state.get("execution_log", [])
    current_log.append("Iniciando análisis CoT con EARIAAgent...")
    try:
        query = state["original_query"]
        doc_paths = state.get("document_paths")
        aspect = state["aspect_to_analyze"]
        k_retrieval = state.get("k_retrieval_for_earia", 5) # Default a 5 si no está en el estado
        force_reprocess_docs = state.get("force_reprocess_docs_for_earia", False) # Default a False

        logger.info(f"EARIAAgent - Query: '{query}', Aspecto: '{aspect}', Docs: {doc_paths}, K: {k_retrieval}, ForceReprocess: {force_reprocess_docs}")

        earia_instance = EARIAAgent() 
        
        cot_result_text_raw = earia_instance.analyze_economic_impact(
            query=query,
            document_paths=doc_paths,
            analysis_type=ANALYSIS_TYPE_COT,
            aspect_to_analyze=aspect,
            k_retrieval=k_retrieval,
            force_reprocess_docs=force_reprocess_docs
        )

        if isinstance(cot_result_text_raw, str) and \
           ("Error:" in cot_result_text_raw and ("Failed to process" in cot_result_text_raw or "Could not get a response" in cot_result_text_raw or "document path does not exist" in cot_result_text_raw or "Falló el procesamiento" in cot_result_text_raw)):
            logger.error(f"EARIA Agent reportó un error: {cot_result_text_raw}")
            current_log.append(f"Error reportado por EARIA Agent: {cot_result_text_raw}")
            return {"error_message": cot_result_text_raw, "earia_cot_result": None, "execution_log": current_log}

        cot_result_text_limpio = limpiar_bloques_think(cot_result_text_raw)
        
        current_log.append(f"Análisis CoT (real) generado por EARIAAgent (limpio): {cot_result_text_limpio[:300]}...")
        return {"earia_cot_result": cot_result_text_limpio, "execution_log": current_log, "error_message": None}

    except Exception as e:
        error_msg = f"Excepción crítica en run_earia_agent_node: {type(e).__name__} - {e}"
        logger.error(error_msg, exc_info=True)
        current_log.append(f"Error crítico en nodo EARIA: {error_msg}")
        return {"error_message": error_msg, "earia_cot_result": None, "execution_log": current_log}

def interpret_cot_result_node(state: AgentWorkflowState) -> Dict[str, Any]:
    logger.info("--- Nodo: Interpretando Resultado CoT (Agente 2 - Intérprete) ---")
    current_log = state.get("execution_log", [])
    current_log.append("Iniciando interpretación del resultado CoT...")
    try:
        cot_result_from_agent1 = state.get("earia_cot_result") # Esta es la salida del Agente 1
        original_query = state.get("original_query")

        if state.get("error_message") and not cot_result_from_agent1:
             logger.warning("Saltando interpretación debido a error en el nodo EARIA y no hay resultado CoT.")
             current_log.append("Saltada interpretación por error previo y falta de resultado CoT.")
             return {"execution_log": current_log} 

        if not cot_result_from_agent1:
            error_msg = "No se encontró resultado CoT válido del agente anterior para interpretar."
            logger.error(error_msg)
            current_log.append(f"Error: {error_msg}")
            return {"error_message": error_msg, "interpretation_result": None, "execution_log": current_log}

        # Este es el prompt que ajustamos para el Agente 2
        prompt_interpreter_str = (
            "Eres un revisor experto y estratega económico con alta capacidad crítica y analítica. "
            "Se te ha proporcionado un texto generado por un Agente 1. Este texto incluye:\n"
            "1. Un 'Título General Identificado' para el contexto de los documentos analizados.\n"
            "2. Un 'Resumen Inicial del Contexto' de dichos documentos.\n"
            "3. Un 'Análisis Detallado' (generalmente un Chain-of-Thought o CoT) sobre una consulta específica.\n\n"
            "La consulta original del usuario que guio al Agente 1 fue: '{original_query}'\n\n"
            "--- INICIO TEXTO COMPLETO DEL AGENTE 1 ---\n"
            "{texto_agente_1}\n" # <--- Variable que espera el prompt
            "--- FIN TEXTO COMPLETO DEL AGENTE 1 ---\n\n"
            "Tu tarea es, en español, realizar un META-ANÁLISIS PROFUNDO y VERBOSO de la sección 'Análisis Detallado' proporcionada por el Agente 1. "
            "Utiliza el 'Título General Identificado' y el 'Resumen Inicial del Contexto' para entender el marco general, pero enfoca tu meta-análisis en la profundidad, validez y exhaustividad del 'Análisis Detallado'.\n"
            "Tu meta-análisis debe incluir los siguientes puntos, siendo extenso y detallado en cada uno:\n"
            "1.  **Evaluación de la Cobertura del 'Análisis Detallado':** ¿Qué tan bien aborda el 'Análisis Detallado' la consulta original del usuario y el aspecto específico que se le pidió analizar (si se conoce)? ¿Cubre los puntos clave esperables?\n"
            "2.  **Análisis de Fortalezas del 'Análisis Detallado':** Identifica y explica con detalle al menos 2-3 puntos fuertes o aspectos bien logrados (ej. claridad, uso de evidencia del contexto, profundidad en ciertos puntos).\n"
            "3.  **Identificación de Debilidades o Limitaciones del 'Análisis Detallado':** Señala y argumenta al menos 2-3 debilidades, omisiones importantes, simplificaciones excesivas, o áreas donde el 'Análisis Detallado' carece de profundidad, evidencia o presenta posibles sesgos.\n"
            "4.  **Profundización Sugerida:** Basado en el 'Análisis Detallado', ¿qué preguntas adicionales surgen? ¿Qué áreas o impactos específicos merecerían una exploración aún más profunda que la realizada por el Agente 1?\n"
            "5.  **Conclusión del Meta-Análisis:** Ofrece una conclusión sopesada sobre la calidad general, utilidad y fiabilidad del 'Análisis Detallado' generado por el Agente 1.\n\n"
            "Sé exhaustivo, crítico y constructivo. Proporciona un análisis sustancial y bien argumentado. "
            "Evita incluir meta-comentarios sobre tu propio proceso de pensamiento en la respuesta final. Estructura tu respuesta claramente."
        )
        prompt_interpreter = ChatPromptTemplate.from_template(prompt_interpreter_str)
        
        chain_interpreter = prompt_interpreter | llm_interpreter_agent | StrOutputParser()
        
        logger.info(f"Agente Intérprete - Realizando meta-análisis del resultado CoT (longitud input: {len(cot_result_from_agent1)} caracteres)...")
        
        # CORRECCIÓN CLAVE AQUÍ: usa "texto_agente_1" como clave en el diccionario de invoke
        interpretation_raw = chain_interpreter.invoke({
            "texto_agente_1": cot_result_from_agent1, 
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

# --- 4. Construcción del Grafo ---
workflow_builder = StateGraph(AgentWorkflowState)

workflow_builder.add_node("earia_analysis_step", run_earia_agent_node)
workflow_builder.add_node("interpretation_step", interpret_cot_result_node)

workflow_builder.set_entry_point("earia_analysis_step")
workflow_builder.add_edge("earia_analysis_step", "interpretation_step")
workflow_builder.add_edge("interpretation_step", END)

app = workflow_builder.compile()

# --- 5. Ejecutar el Grafo ---
if __name__ == "__main__":
    logger.info("--- Iniciando Flujo de Agentes con LangGraph (Usando EARIAAgent real) ---")
    
    example_doc_path_input = "./documents/examples/example_resolution.pdf" 

    if not os.path.exists(example_doc_path_input):
        logger.warning(f"El documento de ejemplo no se encuentra en: {os.path.abspath(example_doc_path_input)}")
        document_paths_for_agent = None 
    else:
        document_paths_for_agent = [example_doc_path_input]

    initial_input_state = AgentWorkflowState(
        original_query="Analizar exhaustivamente ... los objetivos principales del documento 'example_resolution.pdf'...", # Sé específico aquí
        document_paths=document_paths_for_agent,
        aspect_to_analyze="objetivos ... del documento 'example_resolution.pdf'", # Y aquí
        k_retrieval_for_earia=8, # Puedes ajustar este valor
        force_reprocess_docs_for_earia=False, # Cambia a True para forzar el reprocesamiento
        earia_cot_result=None, 
        interpretation_result=None,
        error_message=None,
        execution_log=[]
    )

    logger.info(f"Input inicial para el grafo: {initial_input_state}")

    final_configs = {"recursion_limit": 15}
    
    logger.info("\n--- Eventos del Grafo (Streaming para Observación) ---")
    for chunk in app.stream(initial_input_state, config=final_configs):
        for node_name, current_state_after_node in chunk.items():
            logger.info(f"Estado después del Nodo '{node_name}':")
            log_subset = {k: (f"{str(v)[:100]}..." if isinstance(v, str) and len(v) > 100 else v) 
                          for k, v in current_state_after_node.items() if k != "execution_log"}
            logger.info(f"  Estado (parcial): {log_subset}")
            if "execution_log" in current_state_after_node and current_state_after_node["execution_log"]:
                 logger.info(f"  Última entrada del log de ejecución: {current_state_after_node['execution_log'][-1]}")

    logger.info("\n--- Obteniendo Estado Final con app.invoke() para el Resultado Definitivo ---")
    final_state_result = app.invoke(initial_input_state, config=final_configs)

    logger.info("\n--- RAW Estado Final del Grafo (después de invoke) ---")
    if isinstance(final_state_result, dict):
        log_subset_final = {k: (f"{str(v)[:200]}..." if isinstance(v, str) and len(v) > 200 else v) 
                            for k, v in final_state_result.items() if k != "execution_log"}
        logger.info(f"Estado final (parcial): {log_subset_final}")
    else:
        logger.info(f"Estado final (tipo no esperado): {final_state_result}")

    logger.info("\n--- Estado Final del Grafo (Análisis Detallado) ---")
    if final_state_result and isinstance(final_state_result, dict):
        print(f"\nConsulta Original: {final_state_result.get('original_query', 'No disponible.')}")

        print("\nResultado del Análisis EARIA (CoT - Agente 1 - Limpio):")
        print("-------------------------------------------------")
        earia_output = final_state_result.get('earia_cot_result')
        print(earia_output if earia_output is not None else 'No disponible o error previo.')
        print("-------------------------------------------------")

        print("\nResultado del Meta-Análisis (Agente 2 - Limpio):")
        print("-------------------------------------------------")
        interpretation_output = final_state_result.get('interpretation_result')
        print(interpretation_output if interpretation_output is not None else 'No disponible o error previo.')
        print("-------------------------------------------------")
        
        if final_state_result.get('error_message'):
            print(f"\nMENSAJE DE ERROR DURANTE LA EJECUCIÓN: {final_state_result.get('error_message')}")
        
        print("\nLog de Ejecución Completo (del estado final):")
        execution_log_from_state = final_state_result.get('execution_log', [])
        if execution_log_from_state:
            for entry in execution_log_from_state:
                print(f"- {entry}")
        else:
            print("Log de ejecución no disponible en el estado final.")
    else:
        logger.error("El grafo no produjo un estado final válido o interpretable.")

    logger.info("\n--- Fin del Flujo ---")