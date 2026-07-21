import os
import traceback
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("API key loaded:", bool(api_key))
print("SDK version:", getattr(genai, "__version__", "unknown"))

client = genai.Client(api_key=api_key)
selected_model = "gemini-2.0-flash"
print("Selected model:", selected_model)

print("\nAvailable models from SDK:")
try:
    models = client.models.list()
    for model in models:
        print(model)
except Exception:
    traceback.print_exc()

print("\nTrying to initialize selected model...")
try:
    response = client.models.generate_content(model=selected_model, contents="Say hello in one sentence.")
    print("Response:", response.text)
except Exception:
    traceback.print_exc()
