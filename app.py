import streamlit as st

from src.agente import AgenteCostos
from src.herramientas import ULTIMAS_NOTICIAS

LIMITE_PREGUNTAS = 20

st.set_page_config(page_title="Agente de costos de equipos", page_icon="📊")
st.title("📊 Agente de proyección de costos")
st.caption("Pregúntale por la proyección de costos de los equipos, el modelo o el contexto del mercado.")

with st.sidebar:
    st.header("Sobre este agente")
    st.write(
        "Responde preguntas sobre el análisis de costos de equipos: qué materias primas los explican, "
        "la proyección a 6 meses con sus rangos y noticias actuales del mercado."
    )
    st.subheader("Ejemplos de preguntas")
    st.markdown(
        "- ¿Cuánto costará el Equipo 2 en diciembre?\n"
        "- ¿De qué depende el precio del Equipo 1?\n"
        "- ¿Qué está pasando con el petróleo y cómo afecta los costos?"
    )
    st.info(
        "Este agente usa el plan gratuito de Gemini, por lo que algunas respuestas pueden tardar "
        "un poco más de lo normal."
    )


def mostrar_fuentes(fuentes):
    """Muestra las noticias consultadas con sus enlaces originales."""
    if fuentes:
        with st.expander("📰 Fuentes consultadas"):
            for noticia in fuentes:
                st.markdown(f"- [{noticia['titulo']}]({noticia['enlace']}) — {noticia['medio']}")


if "agente" not in st.session_state:
    st.session_state.agente = AgenteCostos()
    st.session_state.mensajes = []

for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["texto"])
        mostrar_fuentes(mensaje.get("fuentes", []))

preguntas_hechas = sum(1 for m in st.session_state.mensajes if m["rol"] == "user")

if preguntas_hechas >= LIMITE_PREGUNTAS:
    st.warning(f"Alcanzaste el límite de {LIMITE_PREGUNTAS} preguntas por sesión. Recarga la página para empezar de nuevo.")
elif pregunta := st.chat_input("Escribe tu pregunta..."):
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        ULTIMAS_NOTICIAS.clear()
        with st.spinner("El agente está analizando tu pregunta..."):
            respuesta = st.session_state.agente.preguntar(pregunta)
        fuentes = list(ULTIMAS_NOTICIAS)
        st.markdown(respuesta)
        mostrar_fuentes(fuentes)

    st.session_state.mensajes.append({"rol": "assistant", "texto": respuesta, "fuentes": fuentes})