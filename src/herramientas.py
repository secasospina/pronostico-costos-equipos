from pathlib import Path
import pandas as pd
import json
from urllib.parse import quote_plus
import feedparser
import httpx

RUTA_PROYECCION = Path(__file__).resolve().parent.parent / "reports" / "proyeccion_costos_equipos.csv"
RUTA_MODELO = Path(__file__).resolve().parent.parent / "reports" / "resultados_modelo.json"
ULTIMAS_NOTICIAS = []

def consultar_proyeccion(equipo: str) -> str:
    """Devuelve la proyección de costo a 6 meses de un equipo ('Equipo 1' o 'Equipo 2'),
    con el valor esperado y el rango (escenarios bajo y alto) de cada mes."""
    proyeccion = pd.read_csv(RUTA_PROYECCION, header=[0, 1], index_col=0)

    if equipo not in proyeccion.columns.get_level_values(0):
        return f"No existe el equipo '{equipo}'. Opciones: Equipo 1, Equipo 2."

    return proyeccion[equipo].to_string()

def consultar_modelo(equipo: str) -> str:
    """Explica de qué materias primas depende el precio de un equipo ('Equipo 1' o 'Equipo 2'):
    el peso de cada una, si es significativa o ruido, el R² y el error de validación."""
    with open(RUTA_MODELO, encoding="utf-8") as archivo:
        resultados = json.load(archivo)

    if equipo not in resultados:
        return f"No existe el equipo '{equipo}'. Opciones: Equipo 1, Equipo 2."

    datos = resultados[equipo]
    lineas = [f"Modelo de regresión del {equipo}:"]
    for materia in ["Price_X", "Price_Y", "Price_Z"]:
        coeficiente = datos["coeficientes"][materia]
        p_valor = datos["p_valores"][materia]
        estado = "influye" if p_valor < 0.05 else "es ruido"
        lineas.append(f"- {materia}: peso {coeficiente}, p-valor {p_valor} → {estado}")
    lineas.append(f"R²: {datos['r2']} | Error de validación (MAPE): {datos['mape_validacion']:.2%}")

    return "\n".join(lineas)

def buscar_contexto_mercado(tema: str, cantidad: int = 5) -> str:
    """Busca en Google News las noticias más recientes sobre un tema de mercado
    (por ejemplo, 'precio del petróleo' o 'materias primas') y devuelve título, medio y fecha."""
    url = f"https://news.google.com/rss/search?q={quote_plus(tema)}&hl=es-419&gl=CO&ceid=CO:es-419"

    try:
        respuesta = httpx.get(url, timeout=10, follow_redirects=True)
        respuesta.raise_for_status()
    except httpx.HTTPError as error:
        return f"No fue posible consultar las noticias en este momento ({error})."

    feed = feedparser.parse(respuesta.content)
    if not feed.entries:
        return f"No se encontraron noticias recientes sobre '{tema}'."

    noticias = []
    for entrada in feed.entries[:cantidad]:
        medio = entrada.get("source", {}).get("title", "Medio desconocido")
        fecha = entrada.get("published", "sin fecha")
        ULTIMAS_NOTICIAS.append({"titulo": entrada.title, "medio": medio, "enlace": entrada.link})
        noticias.append(f"- {entrada.title} ({medio}, {fecha})")

    return f"Noticias recientes sobre '{tema}':\n" + "\n".join(noticias)