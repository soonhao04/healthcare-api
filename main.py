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
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    client = None

# ----------------------------------------
# 2. HELPER FUNCTIONS (PROMPTS)
# ----------------------------------------

def generate_risk_prompt(patient_data):
    # --- KEEPING YOUR ORIGINAL PERSONA ---
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

    # --- KEEPING YOUR EXACT REPORT STRUCTURE ---
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

    # --- CHANGED: Just ask for the text directly, NO JSON ---
    final_instruction = (
        "Format the response using Markdown headers exactly as requested above. "
        "Do NOT use JSON. Do NOT include any introductory text. Start directly with the report title."
    )

    return f"{persona_instruction}\n{data_block}\n{report_logic}\n{final_instruction}"

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

@app.route('/', methods=['GET'])
def home():
    return "Healthcare AI Server is Running!", 200

@app.route('/get-report', methods=['POST'])
def get_report():
    if not client: return '{"error": "Gemini API key not configured."}', 500

    patient_data = request.get_json()

    try:
        # 1. Use the Detailed Prompt (Just without JSON)
        risk_prompt = generate_risk_prompt(patient_data)

        # 2. Call Gemini WITHOUT the "application/json" requirement
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=risk_prompt,
        )

        # 3. Return the text directly (Perfect for your App)
        return response.text, 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-diet', methods=['POST'])
def generate_diet():
    if not client: return '{"error": "Gemini API key not configured."}', 500

    data = request.get_json()

    try:
        diet_prompt = generate_diet_prompt(data)

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=diet_prompt,
        )

        return response.text, 200

    except Exception as e:
        print(f"Error generating diet: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
