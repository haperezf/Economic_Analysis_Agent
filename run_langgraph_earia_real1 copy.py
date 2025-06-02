# run_langgraph_earia_real.py

import os
import sys
import logging
import re # Para limpiar los bloques <think>
from typing import TypedDict, List, Optional, Any, Dict # <--- Añadido Dict aquí
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
# Esto cargará variables para EARIAAgent (Agente 1), como OLLAMA_MODEL, VECTOR_DB_PATH, etc.
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
                  f"('C:\\LLM\\earia_colombia_agent\\') o que 'earia_agent' esté correctamente en PYTHONPATH.")
    logger.critical(f"PYTHONPATH actual: {os.getenv('PYTHONPATH')}")
    logger.critical(f"sys.path actual: {sys.path}")
    exit()

# --- 1. Definición del Estado del Grafo ---
class AgentWorkflowState(TypedDict):
    original_query: str
    document_paths: Optional[List[str]]
    aspect_to_analyze: str
    earia_cot_result: Optional[str]      # Salida del Agente 1 (EARIA)
    interpretation_result: Optional[str] # Salida del Agente 2 (Intérprete)
    error_message: Optional[str]
    execution_log: List[str]

# --- 2. Configuración de los LLMs de Ollama ---
# El LLM para EARIAAgent se configurará a través de sus propias variables de entorno (OLLAMA_MODEL de .env)
# que su LLMHandler interno leerá.

# Aquí configuramos el LLM para el Agente 2 (Intérprete) con valores FIJOS (hardcodeados).
LLM_AGENT_2_INTERPRETER_MODEL = "gemma3:12b"
OLLAMA_BASE_URL_AGENT2 = "http://localhost:11434"

logger.info(f"Modelo para Agente 2 (Intérprete) FIJADO A: {LLM_AGENT_2_INTERPRETER_MODEL} en {OLLAMA_BASE_URL_AGENT2}")

try:
    llm_interpreter_agent = ChatOllama(
        model=LLM_AGENT_2_INTERPRETER_MODEL, # Usará "phi4-mini-reasoning"
        base_url=OLLAMA_BASE_URL_AGENT2,     # Usará "http://localhost:11434"
        temperature=0.3
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
    # Elimina contenido entre <think> y </think>, incluyendo las etiquetas. re.DOTALL hace que '.' incluya saltos de línea.
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

        logger.info(f"EARIAAgent - Query: '{query}', Aspecto: '{aspect}', Docs: {doc_paths}")

        earia_instance = EARIAAgent() 
        
        k_retrieval = 5 # Puedes hacerlo configurable a través del estado si lo necesitas
        force_reprocess_docs = False # Puedes hacerlo configurable

        cot_result_text_raw = earia_instance.analyze_economic_impact(
            query=query,
            document_paths=doc_paths,
            analysis_type=ANALYSIS_TYPE_COT,
            aspect_to_analyze=aspect,
            k_retrieval=k_retrieval,
            force_reprocess_docs=force_reprocess_docs
        )

        # Verificar si el resultado de EARIA indica un error común
        if isinstance(cot_result_text_raw, str) and \
           ("Error:" in cot_result_text_raw and ("Failed to process" in cot_result_text_raw or "Could not get a response" in cot_result_text_raw or "document path does not exist" in cot_result_text_raw)):
            logger.error(f"EARIA Agent reportó un error: {cot_result_text_raw}")
            current_log.append(f"Error reportado por EARIA Agent: {cot_result_text_raw}")
            return {"error_message": cot_result_text_raw, "earia_cot_result": None, "execution_log": current_log}

        cot_result_text_limpio = limpiar_bloques_think(cot_result_text_raw)
        
        current_log.append(f"Análisis CoT (real) generado por EARIAAgent (limpio): {cot_result_text_limpio[:300]}...")
        return {"earia_cot_result": cot_result_text_limpio, "execution_log": current_log, "error_message": None} # Limpiar error_message si este paso tuvo éxito

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
        cot_result = state.get("earia_cot_result")
        original_query = state.get("original_query")

        # Si hubo un error en el nodo anterior Y no hay resultado CoT, no continuar.
        if state.get("error_message") and not cot_result:
             logger.warning("Saltando interpretación debido a error en el nodo EARIA y no hay resultado CoT.")
             current_log.append("Saltada interpretación por error previo y falta de resultado CoT.")
             # El error_message ya está en el estado, no es necesario devolverlo de nuevo a menos que este nodo genere uno nuevo.
             return {"execution_log": current_log} 

        if not cot_result: # Si no hubo error pero el resultado CoT es None o vacío por alguna razón
            error_msg = "No se encontró resultado CoT válido del agente anterior para interpretar."
            logger.error(error_msg)
            current_log.append(f"Error: {error_msg}")
            return {"error_message": error_msg, "interpretation_result": None, "execution_log": current_log}

        prompt_interpreter_str = (
        "Eres un revisor experto y estratega económico con alta capacidad crítica. "
        "El siguiente texto es un análisis de impacto económico (Análisis Original) generado por otro sistema sobre la consulta: '{original_query}'.\n\n"
        "--- INICIO ANÁLISIS ORIGINAL ---\n"
        "{analisis_cot}\n"
        "--- FIN ANÁLISIS ORIGINAL ---\n\n"
        "Tu tarea es, en español, realizar un meta-análisis PROFUNDO y VERBOSO del Análisis Original. Tu respuesta debe incluir:\n"
        "1.  **Fortalezas del Análisis Original:** Identifica y explica al menos 2-3 puntos fuertes o aspectos bien logrados del análisis proporcionado.\n"
        "2.  **Debilidades o Limitaciones:** Identifica y explica al menos 2-3 debilidades, omisiones, o áreas donde el análisis original podría ser más profundo o necesitaría información adicional (según el propio análisis).\n"
        "3.  **Suposiciones Implícitas:** ¿Detectas alguna suposición importante que el Análisis Original podría estar haciendo sin declararla explícitamente?\n"
        "4.  **Profundización en un Aspecto Clave:** Selecciona el impacto económico más significativo o interesante mencionado en el Análisis Original. Expande sobre este impacto, ofreciendo una perspectiva más detallada sobre sus posibles consecuencias a largo plazo, efectos de segundo orden, o consideraciones estratégicas para las partes interesadas.\n"
        "5.  **Conclusión del Meta-Análisis:** Ofrece una breve conclusión sobre la utilidad general y la robustez del Análisis Original.\n\n"
        "Sé detallado y crítico en tu respuesta. Proporciona un análisis sustancial para cada uno de los puntos solicitados. "
        "Evita incluir meta-comentarios sobre tu proceso de pensamiento en la respuesta final. Estructura tu respuesta claramente."
        )
        
        prompt_interpreter = ChatPromptTemplate.from_template(prompt_interpreter_str)
        
        chain_interpreter = prompt_interpreter | llm_interpreter_agent | StrOutputParser()
        
        logger.info(f"Agente Intérprete - Interpretando análisis CoT (longitud: {len(cot_result)} caracteres)...")
        interpretation_raw = chain_interpreter.invoke({
            "analisis_cot": cot_result,
            "original_query": original_query
        })
        interpretation_limpia = limpiar_bloques_think(interpretation_raw)

        current_log.append(f"Interpretación generada (limpia): {interpretation_limpia[:300]}...")
        return {"interpretation_result": interpretation_limpia, "execution_log": current_log, "error_message": None} # Limpiar error_message si este paso tuvo éxito

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
    
    # Asegúrate de que la ruta del documento sea correcta y accesible desde donde ejecutas este script.
    example_doc_path_input = "./documents/examples/example_resolution.pdf" 

    if not os.path.exists(example_doc_path_input):
        logger.warning(f"El documento de ejemplo no se encuentra en: {os.path.abspath(example_doc_path_input)}")
        logger.warning("EARIAAgent podría no funcionar correctamente si el documento es esencial y no se encuentra.")
        # Puedes decidir si continuar o no. Si el documento es opcional o la KB ya está poblada:
        # document_paths_for_agent = None
        document_paths_for_agent = [example_doc_path_input] # Proceder, EARIAAgent manejará el error de archivo si es crítico.
    else:
        document_paths_for_agent = [example_doc_path_input]


    initial_input_state = AgentWorkflowState(
        original_query="Resumir los principales puntos del documento.",
        document_paths=document_paths_for_agent,
        aspect_to_analyze="objetivos principales y especificos e impacto en los operadores",
        earia_cot_result=None, 
        interpretation_result=None,
        error_message=None,
        execution_log=[]
    )

    logger.info(f"Input inicial para el grafo: {initial_input_state}")

    final_configs = {"recursion_limit": 10} # Configuración para la ejecución
    
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
        # No imprimas el execution_log completo aquí si es muy largo, se imprimirá después.
    else:
        logger.info(f"Estado final (tipo no esperado): {final_state_result}")


    logger.info("\n--- Estado Final del Grafo (Análisis Detallado) ---")
    if final_state_result and isinstance(final_state_result, dict):
        print(f"Consulta Original: {final_state_result.get('original_query', 'No disponible.')}")

        print("\nResultado del Análisis EARIA (CoT - Agente 1 - Limpio):")
        print("-------------------------------------------------")
        earia_output = final_state_result.get('earia_cot_result')
        print(earia_output if earia_output is not None else 'No disponible o error previo.') # Check for None
        print("-------------------------------------------------")

        print("\nResultado de la Interpretación (Agente 2 - Limpio):")
        print("-------------------------------------------------")
        interpretation_output = final_state_result.get('interpretation_result')
        print(interpretation_output if interpretation_output is not None else 'No disponible o error previo.') # Check for None
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
        logger.error("El grafo no produjo un estado final válido o interpretable (final_state_result no es un diccionario o es None).")

    logger.info("\n--- Fin del Flujo ---")