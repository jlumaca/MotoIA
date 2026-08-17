import os

from dotenv import load_dotenv
from openai import OpenAI


# Cargar las variables del archivo .env
load_dotenv()

# Obtener la API Key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: No se encontró OPENAI_API_KEY en el archivo .env")
    exit()


# Crear el cliente de OpenAI
client = OpenAI(api_key=api_key)


# Realizar una consulta
response = client.responses.create(
    model="gpt-5.6-luna",
    input="Explicá en una oración qué es una motocicleta."
)


# Mostrar la respuesta
print("\nRespuesta de la IA:")
print(response.output_text)