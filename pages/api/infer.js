/**
 * pages/api/infer.js
 * ------------------
 * Vercel serverless function (Next.js API route).
 * Accepts a POST with { imageBase64, task, onsetHistory }
 * Calls OpenRouter (google/gemini-2.5-flash) and returns structured JSON.
 */

const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";
const MODEL_CASCADE = [
  "google/gemini-2.5-flash",
  "google/gemini-2.5-flash-lite",
  "openai/gpt-4o-mini",
];

function buildPrompt(task) {
  const isJaundice = task === "jaundice";
  return `You are an expert ophthalmological AI triage system.
Analyze the provided eye image to detect ${isJaundice ? "scleral icterus (jaundice)" : "conjunctival pallor (anemia)"}.

CRITICAL INSTRUCTIONS:
1. Focus ONLY on the ${isJaundice ? "sclera (white of the eye)" : "inner eyelid conjunctiva (lower lid)"}.
2. Completely IGNORE surrounding skin tone, eyelashes, and camera flash glare.
3. Perform a visual heuristic analysis — do NOT attempt colorimetric math.

Return ONLY valid JSON in this EXACT format (no markdown, no text outside JSON):
{
  "sclera_color_assessment": "Describe tissue appearance (e.g. '${isJaundice ? "Clear white, Mild yellowing, Severe yellowing" : "Normal pink-red, Pale pink, Very pale/white"}' )",
  "risk_level": "LOW",
  "confidence_score": 85,
  "clinical_reasoning": "Explain your reasoning, noting any lighting or skin artifacts considered."
}

Rules:
- risk_level must be exactly one of: LOW, MEDIUM, HIGH
- confidence_score must be an integer 0-100
- Do not include any text outside the JSON object`;
}

async function callOpenRouter(apiKey, model, imageBase64, prompt) {
  const response = await fetch(`${OPENROUTER_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://github.com/udhayan26-hub/Hemo-sclera",
      "X-Title": "Hemo-Sclera Diagnostic",
    },
    body: JSON.stringify({
      model,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            {
              type: "image_url",
              image_url: { url: `data:image/jpeg;base64,${imageBase64}` },
            },
          ],
        },
      ],
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw Object.assign(new Error(`OpenRouter ${response.status}: ${errorBody}`), {
      status: response.status,
    });
  }

  const data = await response.json();
  return JSON.parse(data.choices[0].message.content);
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const apiKey =
    process.env.OPENROUTER_API_KEY || process.env.GEMINI_API_KEY;

  if (!apiKey) {
    return res.status(500).json({
      error: true,
      error_detail:
        "OPENROUTER_API_KEY is not configured. Add it to Vercel Environment Variables.",
    });
  }

  const { imageBase64, task = "jaundice", onsetHistory = "" } = req.body;

  if (!imageBase64) {
    return res.status(400).json({ error: true, error_detail: "No image provided." });
  }

  const prompt = buildPrompt(task);
  let lastError = "Unknown error";

  for (const model of MODEL_CASCADE) {
    try {
      console.log(`[inference] Trying model: ${model}`);
      const result = await callOpenRouter(apiKey, model, imageBase64, prompt);

      // Validate required keys
      const required = ["sclera_color_assessment", "risk_level", "confidence_score", "clinical_reasoning"];
      const missing = required.filter((k) => !(k in result));
      if (missing.length > 0) {
        throw new Error(`Model response missing keys: ${missing.join(", ")}`);
      }

      console.log(`[inference] ✅ Success with ${model}`);
      return res.status(200).json({ ...result, _model_used: model });
    } catch (err) {
      lastError = err.message;
      console.error(`[inference] ❌ ${model} failed: ${lastError}`);

      // On 401/403 (bad key) — don't try fallbacks, it won't help
      if (err.status === 401 || err.status === 403) {
        return res.status(401).json({
          error: true,
          error_detail: `Invalid or unauthorized API key. Check OPENROUTER_API_KEY. (${lastError})`,
        });
      }
      // On 404 — try next model
      // On other errors — try next model
      continue;
    }
  }

  return res.status(502).json({
    error: true,
    error_detail: `All models failed. Last error: ${lastError}`,
  });
}

// Increase body size limit for image uploads (default 1MB is too small)
export const config = {
  api: {
    bodyParser: {
      sizeLimit: "10mb",
    },
  },
};
