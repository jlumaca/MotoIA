import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# Cargar variables del archivo .env
load_dotenv()

# Obtener token de Hugging Face
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("ERROR: No se encontró HF_TOKEN en el archivo .env")
    exit()


# Crear cliente de Hugging Face
client = InferenceClient(
    api_key=hf_token
)


# Prompt para generar la imagen
prompt = """
Technical educational illustration of a motorcycle
front brake system, showing the brake disc, brake caliper
and brake pads, clean mechanical workshop style,
realistic technical illustration, detailed,
white background.
"""


print("Generando imagen...")
print("Esto puede tardar unos segundos.")


# Generar imagen
image = client.text_to_image(
    prompt,
    model="stabilityai/stable-diffusion-3-medium-diffusers"
)


# Guardar imagen
image.save("prueba_stable_diffusion.png")


print("Imagen generada correctamente.")
print("Archivo: prueba_stable_diffusion.png")