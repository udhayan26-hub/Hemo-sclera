import google.generativeai as genai
from PIL import Image
import json
import time

# --- NEURAL CONFIGURATION ---
# API Key provided by user
API_KEY = "AIzaSyAhBGkxY8PYKdZ0aKFQH9fEqdLqAugXEXE"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    model_name='gemini-3-flash-preview',
    generation_config={"response_mime_type": "application/json"}
)

def execute_neural_inference(image_path, timeout=3):
    """
    Strategic Decision-Level Fusion Layer.
    Uses multimodal transformers to account for non-linear chromatic artifacts.
    """
    try:
        # Load image via standard Pillow method
        img = Image.open(image_path)
        
        prompt = """
        Perform a professional multimodal analysis of this ocular tissue image. 
        Focus on Conjunctival Pallor (Whiteness/Anemia) and Scleral Icterus (Yellowing/Jaundice).
        The current local math models are struggling with lighting/skin-tone interference.
        Provide a high-accuracy clinical triage outcome.
        Return ONLY a JSON object:
        {"jaundice_risk": "High" or "Low", "anemia_risk": "High" or "Low", "conf": 0.98, "reason": "Short clinical note observing color and tissue markers."}
        """
        
        # Flash is fast, but we need to stay within the demo timeout
        response = model.generate_content([prompt, img])
        
        # DEBUG LOGGING (Visible in Terminal)
        raw_text = response.text.strip()
        print(f"\n[DEBUG] RAW Neural Response: {raw_text}\n")
        
        # Robust JSON extraction: Find the first '{' and last '}'
        import re
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            res_text = match.group(0)
            return json.loads(res_text)
        else:
            raise ValueError("No JSON object found in response.")
        
    except Exception as e:
        # Silent Fallback to Local Baseline if internet fails or API errors
        print(f"[ERROR] Neural Engine Failure: {str(e)}")
        return {
            "jaundice_risk": "Error",
            "anemia_risk": "Error",
            "conf": 0.0, 
            "reason": f"System Alert (Neural Bypass Active): {str(e)}"
        }

if __name__ == "__main__":
    print("Neural Chromatic Ensemble Module Loaded.")
