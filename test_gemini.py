import google.generativeai as genai
import os
import json

# Use the key from inference.py
API_KEY = "AIzaSyAhBGkxY8PYKdZ0aKFQH9fEqdLqAugXEXE"
genai.configure(api_key=API_KEY)

try:
    print("Testing Gemini model listing...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Found model: {m.name}")
            
    print("\nTesting simple text generation...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello! Are you operational?")
    print(f"Response: {response.text}")
    print("\n✅ API KEY AND CONNECTION ARE OK.")
except Exception as e:
    print(f"\n❌ API TEST FAILED: {str(e)}")
