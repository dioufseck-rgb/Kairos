import os
from google import genai

key = os.getenv("GEMINI_KEY")
print("KEY set:", bool(key), "len:", 0 if not key else len(key))

client = genai.Client(api_key=key)
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say 'ok' and nothing else."
)
print(resp.text)
