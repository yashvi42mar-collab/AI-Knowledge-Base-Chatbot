import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from the project .env file.
load_dotenv()

# Check whether the API key was loaded from the environment.
api_key = os.getenv("GEMINI_API_KEY")
print("API key loaded:", bool(api_key))
if not api_key:
    raise SystemExit("GEMINI_API_KEY is missing. Add it to the .env file.")

# Use the current Google GenAI SDK and a supported Flash model with fallback.
client = genai.Client(api_key=api_key)
for model_name in ("gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"):
    try:
        response = client.models.generate_content(model=model_name, contents="Say hello in one sentence.")
        print("Model used:", model_name)
        print(response.text)
        break
    except Exception as exc:
        error_text = str(exc).lower()
        print("Complete error:", exc)
        if "quota" in error_text or "rate limit" in error_text:
            print("Meaning: your API quota or rate limit was exceeded.")
        elif "not found" in error_text or "unsupported" in error_text or "404" in error_text:
            print("Meaning: this model is not available for your account or endpoint.")
        else:
            print("Meaning: the request failed for another API or network reason.")
        if "api key" in error_text or "authentication" in error_text or "forbidden" in error_text:
            raise
        print("Trying the next supported model...")
