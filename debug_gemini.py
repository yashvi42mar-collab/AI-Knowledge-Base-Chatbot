import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("Key loaded:", bool(api_key), "| length:", len(api_key) if api_key else 0)

from google import genai

client = genai.Client(api_key=api_key)

print("\n--- Step 1: Listing models your key can see ---")
try:
    models = list(client.models.list())
    print(f"Found {len(models)} models total")
    for m in models:
        methods = getattr(m, "supported_generation_methods", None) or getattr(m, "supported_actions", None)
        print(f"  {m.name}  |  methods: {methods}")
except Exception as e:
    print("list() FAILED:", repr(e))

print("\n--- Step 2: Trying a direct generate_content call ---")
try:
    resp = client.models.generate_content(model="gemini-2.5-flash", contents="Hello")
    print("SUCCESS:", resp.text)
except Exception as e:
    print("generate_content FAILED:", repr(e))
