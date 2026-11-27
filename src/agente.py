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
- El precio del Equipo 1 depende de las materias primas X (peso 0,2) e Y (peso 0,8). Z es ruido para el Equipo 1.
- El precio del Equipo 2 depende de X, Y y Z por igual (peso de aproximadamente 0,33 cada una). En el Equipo 2 ninguna materia prima es ruido.
- El histórico va de enero de 2010 a agosto de 2023. La proyección cubre de septiembre de 2023
  a febrero de 2024 (6 meses), usando precios reales de X y proyecciones de Y y Z.
- El comportamiento de X coincide con el del petróleo en eventos como la caída de 2014-2016 y la de 2020,
  pero el caso no confirma qué es X: preséntalo como una coincidencia, no como un hecho.

Reglas:
- Usa las herramientas para responder con datos; no inventes cifras.
- Cuando des una proyección, menciona el valor esperado y el rango (escenarios bajo y alto).
- Si te preguntan por el contexto del mercado, busca noticias y combínalas con la proyección,
  citando el medio de cada noticia.
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