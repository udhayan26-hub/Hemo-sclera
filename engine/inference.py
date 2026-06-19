import google.generativeai as genai
from PIL import Image
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- NEURAL CONFIGURATION ---
# API Key fetched securely from environment
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    generation_config={"response_mime_type": "application/json"}
)

def execute_neural_inference(image_input, timeout=5):
    """
    Pure Neural Inference Layer.
    Uses multimodal visual reasoning to bypass local lighting/skin artifacts.
    """
    try:
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = image_input
        
        prompt = """
        You are an expert ophthalmological AI triage system. 
        Analyze the provided image of a patient's eye. Your task is to detect scleral icterus (jaundice) or conjunctival pallor (anemia).

        CRITICAL INSTRUCTIONS:
        1. Focus ONLY on the relevant biological tissue (Sclera for Jaundice, Eyelid for Anemia).
        2. Completely IGNORE the surrounding skin, eyelashes, and camera flash glare.
        3. Do not attempt to calculate exact colorimetric math. Instead, perform a visual heuristic analysis.

        Output your diagnosis strictly in the following JSON format:
        {
          "sclera_color_assessment": "Describe what you see in the tissue (e.g., 'Clear white', 'Mild yellowing', 'Severe yellowing', 'Paler than normal').",
          "risk_level": "LOW", "MEDIUM", or "HIGH",
          "confidence_score": 0-100,
          "clinical_reasoning": "Explain why you chose this risk level, specifically noting how you accounted for lighting or skin artifacts."
        }
        """
        
        # --- CRITICAL FIX 2: FORCE STRICT JSON ---
        response = model.generate_content(
            [prompt, img],
            generation_config={"response_mime_type": "application/json"}
        )
        
        raw_text = response.text.strip()
        print(f"\n[DEBUG] RAW Neural Response: {raw_text}\n")
        
        # Parse the pure JSON response
        return json.loads(raw_text)
        
    except Exception as e:
        # --- CRITICAL FIX 3: EXPOSE THE ERROR ---
        error_msg = f"Neural Engine Failure: {str(e)}"
        print(f"[ERROR] {error_msg}")
        if 'response' in locals():
            print(f"[DEBUG] Raw response was: {response.text}")
            
        return {
            "error": True,
            "error_detail": str(e),
            "sclera_color_assessment": "Error during inference.",
            "clinical_reasoning": f"System Alert: {str(e)}",
            "risk_level": "UNKNOWN"
        }

if __name__ == "__main__":
    print("Neural Chromatic Ensemble Module Loaded.")
