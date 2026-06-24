# API Reference Documentation

## Next.js API Routes

### 1. `/api/infer`
Perform diagnostic inference using a cascading LLM approach.

- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "task": "anemia",
    "features": {
      "r_mean": 120,
      "g_mean": 90,
      "b_mean": 80
    }
  }
  ```
- **Response (Success)**:
  ```json
  {
    "status": "success",
    "model": "google/gemini-2.5-flash",
    "prediction": "Anemia detected (High Risk)",
    "recommendations": ["Consult a medical professional for a complete blood count."]
  }
  ```

## Python Engine Modules

### `engine.color_logic`
Functions for color space transformation and region-of-interest analysis.

- `rgb_to_lab(image)`: Convert RGB image to CIE L*a*b* space.
- `extract_physiological_signals(roi)`: Calculate statistical color metrics across `a*` and `b*` channels.
