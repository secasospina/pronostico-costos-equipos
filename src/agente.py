import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from src.herramientas import buscar_contexto_mercado, consultar_modelo, consultar_proyeccion

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODELOS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
ERRORES_TEMPORALES = {429, 500, 503}

INSTRUCCIONES = """
Eres un consultor de datos que presenta los resultados de un análisis de costos de equipos
para una empresa constructora. Responde siempre en español, de forma clara y breve.

Contexto del análisis:
- Equipo 1: la receta es aproximadamente 0,2·X + 0,8·Y. Z es ruido para el Equipo 1. Como Y es mucho más cara
  que X, Y aporta cerca del 97% del precio y X cerca del 3%.
- Equipo 2: la receta es aproximadamente un promedio simple de X, Y y Z (coeficiente cercano a 0,33 cada una) y
  ninguna es ruido. Por sus precios, Z aporta cerca del 76% del precio, Y cerca del 21% y X cerca del 3%.
- No confundas el coeficiente con el porcentaje del precio: usa la herramienta consultar_modelo para dar ambos.
- El histórico va de enero de 2010 a agosto de 2023. La proyección cubre de septiembre de 2023
  a febrero de 2024 (6 meses), con X, Y y Z proyectadas. El escenario bajo y el alto suponen que las tres
  materias primas llegan a su extremo al mismo tiempo, así que son conservadores.
- Los máximos y mínimos de X coinciden con los del petróleo Brent, y el máximo de Z con el récord del aluminio
  en marzo de 2022. El caso no confirma qué son: preséntalo como una hipótesis, no como un hecho.

Reglas:
- Usa las herramientas para responder con datos; no inventes cifras.
- Cuando des una proyección, menciona el valor esperado y el rango (escenarios bajo y alto).
- Si te preguntan por el contexto del mercado, busca noticias y combínalas con la proyección.
  Cita cada noticia por el nombre del medio, sin enlaces: la aplicación muestra las fuentes aparte.
- Las noticias son actuales, pero la proyección corresponde a septiembre de 2023 a febrero de 2024.
  Aclara esta diferencia y explica que, en un uso real, el modelo se actualizaría con los datos más recientes.
- Sé honesto sobre la incertidumbre: los rangos se abren con los meses.
"""


class AgenteCostos:
    """Agente que responde preguntas sobre la proyección de costos usando Gemini y herramientas propias."""

    def __init__(self):
        self.cliente = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.configuracion = types.GenerateContentConfig(
            system_instruction=INSTRUCCIONES,
            tools=[consultar_proyeccion, consultar_modelo, buscar_contexto_mercado],
        )
        self.historial = []

    def preguntar(self, mensaje: str) -> str:
        """Envía una pregunta al agente, con reintentos y modelo de respaldo si Google está saturado."""
        for modelo in MODELOS:
            for intento in range(3):
                try:
                    chat = self.cliente.chats.create(model=modelo, config=self.configuracion, history=self.historial)
                    respuesta = chat.send_message(mensaje)
                    self.historial = chat.get_history()
                    return respuesta.text
                except errors.APIError as error:
                    if error.code not in ERRORES_TEMPORALES:
                        raise
                    time.sleep(2 ** intento)

        return "El servicio de Gemini está saturado en este momento. Por favor, intenta de nuevo en unos minutos."