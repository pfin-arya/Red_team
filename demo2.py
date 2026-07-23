import requests
import pandas as pd
import json
from openai import AzureOpenAI

# ====================================================
# CONFIGURATION
# ====================================================

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
EXPECTED_VERSION = ""
EXPECTED_MODEL = ""

# =====================================================
# LLM #3 : EVALUATOR
# =====================================================

EVAL_ENDPOINT = ""
EVAL_KEY = ""
EVAL_VERSION = ""
EVAL_MODEL = ""



# ====================================================
# CLIENT CREATION
# ====================================================

def create_client(endpoint, key, version):

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=version
    )


# ====================================================
# GENERATE INITIAL QUESTION
# ====================================================

def generate_seed_question(topic):

    client = create_client(
        GENERATOR_ENDPOINT,
        GENERATOR_KEY,
        GENERATOR_VERSION
    )

    prompt = """
You are generating test cases for an RBI Regulatory Query API.

The API specializes in:

- RBI Master Directions
- RBI Circulars
- RBI Notifications
- Supervisory Returns
- Regulatory Compliance
- Outsourcing Guidelines
- KYC Directions
- AML Requirements
- Digital Lending
- Cyber Security Guidelines
- NBFC Regulations

Generate ONE realistic user question
that a compliance officer would ask.

Topic:

""" + topic + """

Generate the question only from this topic.

Requirements:

- Must belong to RBI regulations
- Must be answerable from RBI circulars
- Must be specific
- Must not be vague

Return ONLY JSON.

Example:

{
    "prompt":"What are RBI requirements for periodic updation of KYC records?"
}
"""

    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content

    content = content.replace("```json", "")
    content = content.replace("```", "").strip()

    return json.loads(content)["prompt"]


# ====================================================
# CALL REGINTEL API
# ====================================================

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

    print("\nAPI STATUS:", response.status_code)

    try:

        data = response.json()

        if isinstance(data, dict):

            if "answer" in data:
                return data["answer"]

            return json.dumps(data)

        return str(data)

    except Exception:

        return response.text


# ====================================================
# GENERATE FOLLOWUPS
# ====================================================

def generate_followups(seed_prompt, api_response):

    client = create_client(
        GENERATOR_ENDPOINT,
        GENERATOR_KEY,
        GENERATOR_VERSION
    )

    prompt = f"""
You are creating test cases for the RBI RegIntel API.

Original Question:

{seed_prompt}

API Response:

{api_response}

Create FOUR follow-up questions.

Requirements:

1. Same RBI topic
2. Increasing complexity
3. Regulatory perspective
4. Compliance perspective
5. Operational perspective
6. Reporting perspective

Do NOT repeat wording.

Return JSON only.

[
 {{"prompt":"..."}},
 {{"prompt":"..."}},
 {{"prompt":"..."}},
 {{"prompt":"..."}}
]
"""

    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content

    content = content.replace("```json", "")
    content = content.replace("```", "").strip()

    data = json.loads(content)

    return [x["prompt"] for x in data]


# ====================================================
# GENERATE EXPECTED ANSWER
# ====================================================

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
                "role": "system",
                "content": """
You are an RBI regulatory expert.

Generate the expected answer that a
Regulatory Intelligence platform should return.

Requirements:

- Focus on key compliance requirements.
- Be concise.
- Do not write textbook explanations.
- Mention major RBI requirements only.
- Generate an answer that is likely to
appear in a regulatory knowledge base.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# ====================================================
# EVALUATE RESPONSE
# ====================================================

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
                "role": "system",
                "content": """
You are evaluating an RBI Regulatory Intelligence API.

Compare:

1. Expected Answer
2. API Response

Return ONLY:

Yes

or

No

Guidelines:

Return YES if:
- The API response addresses the same RBI topic.
- The regulatory intent is preserved.
- The response substantially answers the question.
- Missing minor details is acceptable.
- Different wording is acceptable.

Return NO if:
- The API response discusses a different RBI regulation.
- The API response misses the primary requirement.
- The regulatory meaning is materially different.
- The response is unrelated.

Focus on:
1. Topic Match
2. Regulatory Intent
3. Compliance Meaning

Do not require identical wording.
Do not require identical detail level.

Output only Yes or No.
"""
            },
            {
                "role": "user",
                "content": f"""
Expected:

{expected}

API Response:

{actual}
"""
            }
        ]
    )

    return response.choices[0].message.content.strip()


# ====================================================
# MAIN EXECUTION (50 TEST CASES)
# ====================================================

RBI_TOPICS = [

    "KYC Direction",
    "Customer Due Diligence",
    "Beneficial Ownership",
    "Periodic KYC Updation",
    "AML Requirements",
    "Suspicious Transaction Reporting",
    "Fraud Risk Management",
    "Fraud Reporting",
    "NPA Classification",
    "Asset Provisioning"

]

results = []

ALL_PROMPTS = []

NUM_SEEDS = 10

print(f"\nGenerating {NUM_SEEDS} seed questions ...")

for seed_num in range(NUM_SEEDS):

    try:

        print(f"\nSeed {seed_num + 1}")

        topic = RBI_TOPICS[
        seed_num % len(RBI_TOPICS)
        ]

        seed_question = generate_seed_question(topic)

        print(seed_question)

        seed_response = call_regintel_api(
            seed_question
        )

        followups = generate_followups(
            seed_question,
            seed_response
        )

        current_group = [
            seed_question
        ] + followups

        ALL_PROMPTS.extend(current_group)

    except Exception as e:

        print(
            f"Failed generating seed group {seed_num + 1}: {e}"
        )

print(
    f"\nTotal Prompts Generated = {len(ALL_PROMPTS)}"
)

# Safety check

ALL_PROMPTS = ALL_PROMPTS[:50]

print(
    f"Final Test Cases Count = {len(ALL_PROMPTS)}"
)

# ====================================================
# RUN ALL 50 TEST CASES
# ====================================================

for idx, prompt in enumerate(ALL_PROMPTS):

    print(
        f"\nRunning Test Case {idx + 1} / {len(ALL_PROMPTS)}"
    )

    # ---------------------------------
    # Generate Expected Answer
    # ---------------------------------

    try:

        expected_output = generate_expected_answer(
            prompt
        )

    except Exception as e:

        expected_output = (
            f"EXPECTED_ERROR: {str(e)}"
        )

    # ---------------------------------
    # Call API
    # ---------------------------------

    try:

        api_response = call_regintel_api(
            prompt
        )

    except Exception as e:

        api_response = (
            f"API_ERROR: {str(e)}"
        )

    # ---------------------------------
    # Evaluate
    # ---------------------------------

    try:

        comparison = evaluate_response(
            expected_output,
            api_response
        )

    except Exception as e:

        comparison = (
            f"EVAL_ERROR: {str(e)}"
        )

    # ---------------------------------
    # Verdict
    # ---------------------------------

    verdict = comparison.strip().lower()

    if verdict == "yes":

        worked = "Yes"

    elif verdict == "no":

        worked = "No"

    else:

        worked = "Evaluation Failed"

    results.append({

        "Test ID": idx + 1,

        "Prompt": prompt,

        "Expected Output": expected_output,

        "API Response": api_response,

        "Comparison Result": comparison,

        "Worked As Intended": worked

    })

# ====================================================
# REPORT SUMMARY
# ====================================================

total_tests = len(results)

passed = len([
    r
    for r in results
    if r["Worked As Intended"] == "Yes"
])

failed = len([
    r
    for r in results
    if r["Worked As Intended"] == "No"
])

eval_failed = len([
    r
    for r in results
    if r["Worked As Intended"] == "Evaluation Failed"
])

success_rate = round(
    (passed / total_tests) * 100,
    2
) if total_tests > 0 else 0

print("\n==========================")
print("EXECUTION SUMMARY")
print("==========================")

print(f"Total Tests      : {total_tests}")
print(f"Passed           : {passed}")
print(f"Failed           : {failed}")
print(f"EvaluationFailed : {eval_failed}")
print(f"Success Rate     : {success_rate}%")

# ====================================================
# SAVE EXCEL
# ====================================================

df = pd.DataFrame(results)

summary_df = pd.DataFrame([{

    "Total Tests": total_tests,

    "Passed": passed,

    "Failed": failed,

    "Evaluation Failed": eval_failed,

    "Success Rate (%)": success_rate

}])

output_file = (
    "RegIntel_Test_Report_50Cases.xlsx"
)

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    df.to_excel(
        writer,
        sheet_name="Test Results",
        index=False
    )

    summary_df.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

print("\nSUCCESS")
print("Excel Generated:")
print(output_file)