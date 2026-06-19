import google.generativeai as genai
import os
import json

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

try:
    print("Testing Gemini model listing...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Found model: {m.name}")
            
    print("\nTesting simple text generation...")
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Hello! Are you operational?")
    print(f"Response: {response.text}")
    print("\n[OK] API KEY AND CONNECTION ARE OK.")
except Exception as e:
    print(f"\n[FAIL] API TEST FAILED: {str(e)}")
