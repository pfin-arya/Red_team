import requests
import pandas as pd
import logging
import json
from openai import AzureOpenAI

# =====================================================
# CONFIGURATION
# =====================================================

# RBI REGINTEL API
API_URL = ""

API_HEADERS = {
    "Content-Type": "application/json"
}

# =====================================================
# LLM #1 : TEST CASE GENERATOR
# =====================================================

GENERATOR_ENDPOINT = ""
GENERATOR_KEY = ""
GENERATOR_VERSION = ""
GENERATOR_MODEL = ""

# =====================================================
# LLM #2 : EXPECTED ANSWER GENERATOR
# =====================================================

EXPECTED_ENDPOINT = ""
EXPECTED_KEY = ""
EXPECTED_MODEL = ""

# =====================================================
# LLM #3 : EVALUATOR
# =====================================================

EVAL_ENDPOINT = ""
EVAL_KEY = ""
EVAL_MODEL = ""

logging.basicConfig(level=logging.INFO)

# =====================================================
# CREATE CLIENT
# =====================================================

def create_client(endpoint, key, version):

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=version
    )

# =====================================================
# LLM #1
# GENERATE FIRST 10 DIVERSE TEST CASES
# =====================================================

def generate_initial_test_cases():

    client = create_client(
        GENERATOR_ENDPOINT,
        GENERATOR_KEY,
        GENERATOR_VERSION
    )

    prompt = """
Generate 10 completely different RBI regulatory questions.

Cover:
1. KYC
2. AML
3. Supervisory Returns
4. Cyber Security
5. Outsourcing
6. Digital Lending
7. NBFC
8. UCB
9. Payment Systems
10. Reporting Requirements

Return only JSON array.

Example:

[
 {"prompt":"Explain RBI KYC Master Direction"},
 {"prompt":"Latest cyber security guidelines"}
]
"""

    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.9
    )

    return json.loads(
        response.choices[0].message.content
    )

# =====================================================
# EXPAND EACH TEST CASE
# =====================================================

def generate_followups(prompt, api_response):

    client = create_client(
        GENERATOR_ENDPOINT,
        GENERATOR_KEY,
        GENERATOR_VERSION
    )

    expansion_prompt = f"""
Prompt:
{prompt}

API Response:
{api_response}

Generate 4 NEW questions.

Requirements:

- Related to the above topic
- Increasing complexity
- Different wording
- Different regulatory angle

Return JSON:

[
 {{ "prompt":"..." }},
 {{ "prompt":"..." }}
]
"""

    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {
                "role":"user",
                "content": expansion_prompt
            }
        ],
        temperature=0.8
    )

    return json.loads(
        response.choices[0].message.content
    )

# =====================================================
# CALL REGINTEL API
# =====================================================

def call_regintel_api(prompt):

    payload = {
        "Query": prompt
    }

    response = requests.post(
        API_URL,
        headers=API_HEADERS,
        json=payload,
        verify=False,
        timeout=60
    )

    try:
        data = response.json()

        if "answer" in data:
            return data["answer"]

        return str(data)

    except:
        return response.text

# =====================================================
# EXPECTED ANSWER GENERATOR
# =====================================================

def generate_expected_answer(prompt):

    client = create_client(
        EXPECTED_ENDPOINT,
        EXPECTED_KEY,
        EXPECTED_VERSION
    )

    response = client.chat.completions.create(
        model=EXPECTED_MODEL,
        messages=[
            {
                "role":"system",
                "content":
                """
                You are an RBI compliance expert.

                Generate the ideal expected answer
                for the given question.
                """
            },
            {
                "role":"user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

# =====================================================
# EVALUATOR
# =====================================================

def evaluate_response(expected, actual):

    client = create_client(
        EVAL_ENDPOINT,
        EVAL_KEY,
        EVAL_VERSION
    )

    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[
            {
                "role":"system",
                "content":
                """
                Compare expected answer and API answer.

                If they substantially match in meaning:
                Yes

                Else:
                No

                Return only Yes or No.
                """
            },
            {
                "role":"user",
                "content":
                f"""
Expected:
{expected}

API Response:
{actual}
"""
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# =====================================================
# MAIN EXECUTION
# =====================================================

results = []

print("Generating 10 seed test cases...")

seed_cases = generate_initial_test_cases()

all_prompts = []

for item in seed_cases:

    prompt = item["prompt"]

    api_answer = call_regintel_api(prompt)

    all_prompts.append(prompt)

    followups = generate_followups(
        prompt,
        api_answer
    )

    for q in followups:

        all_prompts.append(q["prompt"])

# Safety check
all_prompts = all_prompts[:50]

print(f"Total prompts: {len(all_prompts)}")

# =====================================================
# PROCESS ALL 50 TEST CASES
# =====================================================

for idx, prompt in enumerate(all_prompts):

    try:

        print(f"Running test {idx+1}")

        expected_output = generate_expected_answer(
            prompt
        )

        api_response = call_regintel_api(
            prompt
        )

        verdict = evaluate_response(
            expected_output,
            api_response
        )

        results.append({

            "Test ID": idx + 1,

            "Prompt": prompt,

            "Expected Output": expected_output,

            "API Response": api_response,

            "Comparison Result": verdict,

            "Worked As Intended":
                "Yes"
                if verdict.lower() == "yes"
                else "No"
        })

    except Exception as e:

        results.append({

            "Test ID": idx + 1,

            "Prompt": prompt,

            "Expected Output": "ERROR",

            "API Response": str(e),

            "Comparison Result": "ERROR",

            "Worked As Intended": "No"
        })

# =====================================================
# SAVE REPORT
# =====================================================

output_df = pd.DataFrame(results)

output_df.to_excel(
    "RegIntel_Automated_Test_Report.xlsx",
    index=False
)

print(
    "Completed. Report saved as "
    "RegIntel_Automated_Test_Report.xlsx"
)