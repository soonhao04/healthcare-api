import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv

# ----------------------------------------
# 1. SETUP & CONFIGURATION
# ----------------------------------------
load_dotenv()

app = Flask(__name__)
CORS(app)

try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    client = None

# ----------------------------------------
# 2. HELPER FUNCTIONS (PROMPTS)
# ----------------------------------------

def generate_risk_prompt(patient_data):
    persona_instruction = (
        "You are an expert Clinical Risk Analyst. Review the patient data and "
        "generate a detailed risk assessment report."
    )

    data_block = f"""
    - Age: {patient_data.get("Age")}
    - Gender: {patient_data.get("Gender")}
    - BMI: {patient_data.get("BMI")}
    - BP Status: {patient_data.get("Blood Pressure Status")}
    - Smoking: {patient_data.get("Smoking Status")}
    - Activity: {patient_data.get("Physical Activity")}
    - Family History: {patient_data.get("Family History of NCDs")}
    """

    # --- UPDATED INSTRUCTION HERE ---
    report_logic = (
        "## Overall NCD Risk Assessment\n"
        " (A single paragraph summarizing the primary risk category. "
        "You MUST bold the specific risk level keyword, i.e., write it as **Low**, **Moderate**, **High**, or **Critical**.)\n"
        "## Key Risk Factors & Interpretation\n"
        " (Bullet points detailing 3-5 specific data points that contribute most to the risk. "
        "Interpret the data, e.g., 'Calculated BMI of 32.5 indicates Obesity.').\n"
        "## Personalized Action Plan\n"
        " (3-5 concrete, actionable, and patient-friendly steps to mitigate the identified risks.)\n"
        "## Next Recommended Steps\n"
        " (A concluding sentence on consulting a primary care physician.)\n"
    )

    output_format_instruction = (
        "--- OUTPUT REQUIREMENTS ---\n"
        "You must return the result strictly as valid JSON with exactly two keys:\n"
        "1. 'risk_level': A single string value. MUST be one of: 'Low', 'Moderate', 'High', 'Critical'.\n"
        "2. 'report_markdown': A string containing the full report formatted using Markdown headings exactly as requested below:\n"
        f"{report_logic}\n"
        "Do not include any text outside the JSON object."
    )

    return f"{persona_instruction}\n{data_block}\n{output_format_instruction}"

def generate_diet_prompt(data):
    calories = data.get('targetCalories', 2000)
    goal = data.get('dietaryGoal', 'Maintain Weight')

    return (
        f"You are a professional Nutritionist. Create a 1-day meal plan for a client "
        f"whose goal is to '{goal}' with a target of {calories} calories per day.\n\n"
        f"Requirements:\n"
        f"1. Provide exactly 3 meals: Breakfast, Lunch, Dinner.\n"
        f"2. Provide 1 Snack option.\n"
        f"3. For each meal, list the Name, approximate Calories, and Macronutrients (Protein, Carbs, Fat).\n"
        f"4. Keep the meals simple and healthy.\n"
        f"5. Format the output in clean Markdown (use bolding for Meal Names).\n"
        f"6. Do NOT include introductory text, just the plan."
    )

# ----------------------------------------
# 3. API ROUTES
# ----------------------------------------

@app.route('/get-report', methods=['POST'])
def get_report():
    if not client: return '{"error": "Gemini API key not configured."}', 500

    patient_data = request.get_json()

    try:
        risk_prompt = generate_risk_prompt(patient_data)

        config = genai.types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json"
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=risk_prompt,
            config=config,
        )

        # --- THE FIX IS HERE ---
        # The AI gives us JSON: {"risk_level": "High", "report_markdown": "..."}
        # We parse it HERE so the mobile app gets just the clean text.
        import json
        result = json.loads(response.text)
        report_text = result.get("report_markdown", "Error: No report text found.")
        
        return report_text, 200
        # -----------------------

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/generate-diet', methods=['POST'])
def generate_diet():
    if not client: return '{"error": "Gemini API key not configured."}', 500

    data = request.get_json()

    try:
        diet_prompt = generate_diet_prompt(data)

        # Note: Diet does NOT return JSON, just plain Markdown text
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=diet_prompt,
        )

        return response.text, 200

    except Exception as e:
        print(f"Error generating diet: {e}")
        return jsonify({"error": str(e)}), 500

# ----------------------------------------
# 4. SERVER START (MUST BE AT THE END)
# ----------------------------------------
if __name__ == '__main__':
    print("Starting Flask server on port 5000...")

    app.run(host='0.0.0.0', port=5000, debug=True)
