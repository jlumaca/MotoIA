import os
import json

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="MotoIA",
    page_icon="🏍️",
    layout="centered"
)

# ============================================================
# ESTILOS VISUALES
# ============================================================

st.markdown(
    """
    <style>

    /* Fondo general */
    .stApp {
        background-color: #f5f7fa;
    }

    /* Título principal */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    /* Subtítulo */
    .main-subtitle {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Header */
    .app-header {
        padding: 20px;
        border-radius: 12px;
        background-color: #1f2937;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Tarjetas */
    .info-card {
        padding: 20px;
        border-radius: 12px;
        background-color: white;
        border: 1px solid #e5e7eb;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777;
        font-size: 14px;
        padding: 25px;
        margin-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONEXIÓN CON OPENAI
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Si no se encuentra mediante variables de entorno,
# intentar obtenerla desde los secrets de Streamlit.
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error(
        "No se encontró la API Key. "
        "Verificá la configuración de los secretos."
    )
    st.stop()

client = OpenAI(api_key=api_key)

# ============================================================
# CONEXIÓN CON HUGGING FACE
# ============================================================

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        hf_token = None

if not hf_token:
    st.error(
        "No se encontró el token de Hugging Face. "
        "Verificá la configuración de los secretos."
    )
    st.stop()

hf_client = InferenceClient(
    api_key=hf_token
)

# ============================================================
# ENCABEZADO
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div style="font-size: 42px;">🏍️</div>
        <h1>MotoIA</h1>
        <p>Asistente inteligente para diagnóstico de motocicletas</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">

    <h3>🤖 ¿Qué es MotoIA?</h3>

    <p>
    MotoIA es una herramienta de Inteligencia Artificial
    diseñada para analizar los síntomas descriptos sobre
    una motocicleta y generar un diagnóstico mecánico
    orientativo.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    """
    ⚠️ El resultado generado por la IA es orientativo.
    No reemplaza la inspección de un mecánico profesional.
    """
)


# ============================================================
# DATOS DE LA MOTOCICLETA
# ============================================================

st.header("🔧 Datos de la motocicleta")

marca = st.text_input(
    "Marca",
    placeholder="Ejemplo: Yamaha"
)

modelo = st.text_input(
    "Modelo",
    placeholder="Ejemplo: YBR 125"
)

anio = st.number_input(
    "Año",
    min_value=1950,
    max_value=2026,
    value=2022
)

kilometraje = st.number_input(
    "Kilometraje",
    min_value=0,
    value=10000,
    step=500
)


# ============================================================
# DESCRIPCIÓN DEL PROBLEMA
# ============================================================

st.header("📝 Describí el problema")

problema = st.text_area(
    "¿Qué le sucede a la motocicleta?",
    placeholder=(
        "Ejemplo: La moto pierde potencia cuando acelero "
        "y algunas veces se apaga cuando está caliente."
    ),
    height=150
)


# ============================================================
# FUNCIÓN DE DIAGNÓSTICO
# ============================================================

def analizar_motocicleta(
    marca,
    modelo,
    anio,
    kilometraje,
    problema
):
    
    
    """
    Envía los datos de la motocicleta al modelo de IA
    y obtiene un diagnóstico estructurado.
    """

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
Actúa como un asistente técnico especializado en diagnóstico
preventivo de motocicletas.

Tu función es analizar los síntomas proporcionados por el usuario
y generar un diagnóstico mecánico ORIENTATIVO que pueda servir
como apoyo para el personal de un taller.

IMPORTANTE:

- No afirmes que una falla existe con certeza.
- No reemplaces la inspección física de un mecánico.
- No inventes información que no haya sido proporcionada.
- Si la información es insuficiente, indícalo.
- Priorizá las causas que sean más compatibles con los síntomas.
- Utilizá lenguaje técnico pero comprensible.
- Las comprobaciones recomendadas deben ser razonables y seguras.
- No indiques procedimientos peligrosos para personas sin
  conocimientos mecánicos.

DATOS DE LA MOTOCICLETA:

Marca: {marca}
Modelo: {modelo}
Año: {anio}
Kilometraje: {kilometraje} km

PROBLEMA DESCRIPTO POR EL USUARIO:

{problema}

Analizá la información proporcionada.

La respuesta debe permitir al usuario comprender:

1. Qué problema podría estar relacionado con los síntomas.
2. Cuáles son las causas posibles.
3. Qué elementos debería revisar un mecánico.
4. Qué nivel de urgencia podría tener.
5. Qué información adicional sería útil obtener.

No presentes el diagnóstico como definitivo.
"""


    # --------------------------------------------------------
    # ESQUEMA DE SALIDA
    # --------------------------------------------------------

    schema = {
        "type": "object",
        "properties": {
            "diagnostico": {
                "type": "string"
            },
            "posibles_causas": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "comprobaciones_recomendadas": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "nivel_urgencia": {
                "type": "string",
                "enum": [
                    "BAJO",
                    "MEDIO",
                    "ALTO"
                ]
            },
            "recomendacion_final": {
                "type": "string"
            },
            "informacion_adicional": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": [
            "diagnostico",
            "posibles_causas",
            "comprobaciones_recomendadas",
            "nivel_urgencia",
            "recomendacion_final",
            "informacion_adicional"
        ],
        "additionalProperties": False
    }


    # --------------------------------------------------------
    # CONSULTA A LA IA
    # --------------------------------------------------------

    response = client.responses.create(
        model="gpt-5.6-luna",

        input=prompt,

        text={
            "format": {
                "type": "json_schema",
                "name": "diagnostico_motocicleta",
                "strict": True,
                "schema": schema
            }
        }
    )


    # --------------------------------------------------------
    # CONVERTIR LA RESPUESTA JSON
    # --------------------------------------------------------

    resultado = json.loads(response.output_text)

    return resultado


# ============================================================
# GENERACIÓN DE IMAGEN
# ============================================================

def generar_ilustracion(problema, diagnostico):
    """
    Genera una ilustración técnica relacionada con el
    problema y diagnóstico obtenidos.
    """

    prompt_imagen = f"""
        Create a technical educational illustration of a motorcycle
        mechanical system related to the following diagnostic situation.

        User problem:
        {problema}

        Orientative diagnosis:
        {diagnostico}

        The image must be:
        - A technical mechanical illustration.
        - Educational and clear.
        - Focused on motorcycle components.
        - Suitable for a motorcycle workshop.
        - Realistic but illustrative.
        - On a clean background.
        - Without text labels.
        - Without people.
        - Not presented as photographic evidence of a real failure.

        The image is only an educational visualization and must not
        represent a confirmed mechanical failure.
        """

    imagen = hf_client.text_to_image(
        prompt_imagen,
        model="stabilityai/stable-diffusion-3-medium-diffusers"
    )

    return imagen

# ============================================================
# BOTÓN DE ANÁLISIS
# ============================================================

if st.button(
    "🔍 Analizar problema",
    type="primary"
):

    if not marca or not modelo or not problema:

        st.warning(
            "Por favor, completá la marca, el modelo y "
            "la descripción del problema."
        )

    else:

        with st.spinner(
            "🤖 Analizando el problema..."
        ):

            try:

                resultado = analizar_motocicleta(
                    marca,
                    modelo,
                    anio,
                    kilometraje,
                    problema
                )

            except Exception as error:

                st.error(
                    "Ocurrió un error al comunicarse con "
                    "el modelo de IA."
                )

                st.code(str(error))

                st.stop()


        # ====================================================
        # MOSTRAR RESULTADO
        # ====================================================

        st.divider()

        st.header("🔎 Resultado del análisis")


        # Diagnóstico
        st.subheader("🔧 Diagnóstico orientativo")

        st.write(
            resultado["diagnostico"]
        )


        # Posibles causas
        st.subheader("⚙️ Posibles causas")

        for causa in resultado["posibles_causas"]:

            st.write(
                f"• {causa}"
            )


        # Comprobaciones
        st.subheader("🔍 Comprobaciones recomendadas")

        for comprobacion in resultado[
            "comprobaciones_recomendadas"
        ]:

            st.write(
                f"• {comprobacion}"
            )


        # Urgencia
        st.subheader("🚦 Nivel de urgencia")

        urgencia = resultado[
            "nivel_urgencia"
        ]

        if urgencia == "ALTO":

            st.error(
                f"Nivel de urgencia: {urgencia}"
            )

        elif urgencia == "MEDIO":

            st.warning(
                f"Nivel de urgencia: {urgencia}"
            )

        else:

            st.success(
                f"Nivel de urgencia: {urgencia}"
            )


        # Recomendación
        st.subheader("📋 Recomendación final")

        st.write(
            resultado["recomendacion_final"]
        )


        # Información adicional
        st.subheader(
            "❓ Información adicional necesaria"
        )

        for informacion in resultado[
            "informacion_adicional"
        ]:

            st.write(
                f"• {informacion}"
            )

        # ============================================================
        # ILUSTRACIÓN GENERADA POR IA
        # ============================================================

        st.divider()

        st.subheader("🎨 Visualización del diagnóstico")

        st.write(
            """
            La siguiente imagen fue generada mediante un modelo
            de generación de imágenes a partir del problema
            y diagnóstico obtenidos.
            
            Su finalidad es únicamente educativa y orientativa.
            """
        )

        with st.spinner("🎨 Generando ilustración..."):

            try:

                imagen = generar_ilustracion(
                    problema,
                    resultado["diagnostico"]
                )

                st.image(
                    imagen,
                    caption="Ilustración generada mediante Stable Diffusion"
                )

            except Exception as error:

                st.warning(
                    "No fue posible generar la ilustración."
                )

                st.code(str(error))


# ============================================================
# CÓMO FUNCIONA
# ============================================================

st.divider()

st.header("ℹ️ ¿Cómo funciona MotoIA?")

st.markdown(
    """
**1. Ingresá los datos de la motocicleta**

Indicá marca, modelo, año y kilometraje.

**2. Describí el problema**

Contá con tus palabras qué comportamiento extraño
presenta la motocicleta.

**3. Analizá el problema**

MotoIA envía la información a un modelo de
Inteligencia Artificial mediante un prompt diseñado
específicamente para el diagnóstico orientativo.

**4. Recibí el resultado**

La aplicación presenta posibles causas,
comprobaciones recomendadas, nivel de urgencia
y recomendaciones.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>🏍️ MotoIA</strong>
        <br>
        Asistente inteligente para talleres de motocicletas
        <br><br>
        Proyecto de Inteligencia Artificial
        <br>
        2026
    </div>
    """,
    unsafe_allow_html=True
)