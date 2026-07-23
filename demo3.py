import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from openai import AzureOpenAI

try:
    from docx import Document  # pip install python-docx
except Exception:
    Document = None

# ====================================================
# CONFIGURATION
# ====================================================

# API runtime config (fill these)
API_URL = ""
API_METHOD = "POST"  # POST / GET
API_TIMEOUT = 60
API_VERIFY_SSL = False  # as requested in reference
API_HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer <token>"
}

# Query payload mapping
# If your API expects a different key than "Query", change this.
API_QUERY_FIELD = "Query"

# Input docs
API_DOC_PATH = r"C:\Users\CINT037\OneDrive - Poonawalla Fincorp Limited\Desktop\Red_team_1\API Documentation - RegIntel.docx"  # e.g. r"C:\path\to\api_documentation.docx"
NUM_SEEDS = 10
FOLLOWUPS_PER_SEED = 4
MAX_TESTS = 50

# Optional topic hints (used only if doc summary is sparse)
FALLBACK_TOPICS = [
    # "KYC Direction",
    # "Customer Due Diligence",
    # "Beneficial Ownership",
    # "Periodic KYC Updation",
    # "AML Requirements",
    # "Suspicious Transaction Reporting",
    # "Fraud Risk Management",
    # "Fraud Reporting",
    # "NPA Classification",
    # "Asset Provisioning",
]

# =====================================================
# LLM #1 : TEST CASE GENERATOR
# =====================================================

GENERATOR_ENDPOINT = ""
GENERATOR_KEY = ""
GENERATOR_VERSION = "2025-01-01-preview"
GENERATOR_MODEL = "gpt-4o"

# =====================================================
# LLM #2 : EXPECTED ANSWER GENERATOR
# =====================================================

EXPECTED_ENDPOINT = ""
EXPECTED_KEY = ""
EXPECTED_VERSION = "2025-01-01-preview"
EXPECTED_MODEL = "gpt-4o"

# =====================================================
# LLM #3 : EVALUATOR
# =====================================================

EVAL_ENDPOINT = ""
EVAL_KEY = ""
EVAL_VERSION = "2025-01-01-preview"
EVAL_MODEL = "gpt-4o"

# ====================================================
# LOGGING
# ====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ====================================================
# CLIENT CREATION
# ====================================================


def create_client(endpoint: str, key: str, version: str) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=version
    )


# ====================================================
# UTILS
# ====================================================


def strip_code_fences(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^```(?:json|python|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_safely(text: str) -> Any:
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
        # Try extracting first JSON object/array block
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
        if match:
            return json.loads(match.group(1))
        raise


def truncate_text(text: str, max_chars: int = 30000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


# ====================================================
# DOCUMENT PARSING
# ====================================================


def markdownize_docx(doc: Document) -> str:
    """Extract paragraphs and tables from DOCX in document order."""
    text_parts: List[str] = []

    for element in doc.element.body:
        if element.tag.endswith("p"):
            para = docx.text.paragraph.Paragraph(element, doc)  # type: ignore[name-defined]
            if para.text.strip():
                text_parts.append(para.text.strip())
                text_parts.append("")
        elif element.tag.endswith("tbl"):
            table = docx.table.Table(element, doc)  # type: ignore[name-defined]
            if table.rows:
                headers = [cell.text.strip() for cell in table.rows[0].cells]
                text_parts.append("| " + " | ".join(headers) + " |")
                text_parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in table.rows[1:]:
                    cells = [cell.text.strip() for cell in row.cells]
                    text_parts.append("| " + " | ".join(cells) + " |")
                text_parts.append("")

    return "\n".join(text_parts).strip()


def load_document_text(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Documentation file not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix in [".md", ".txt", ".log", ".csv", ".yaml", ".yml"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".json":
        raw = path.read_text(encoding="utf-8", errors="ignore")
        obj = json.loads(raw)
        return json.dumps(obj, indent=2, ensure_ascii=False)

    if suffix == ".docx":
        if Document is None:
            raise ImportError("python-docx is required for .docx files. Install: pip install python-docx")
        import docx  # local import for markdownize_docx references
        globals()["docx"] = docx
        doc = Document(file_path)
        return markdownize_docx(doc)

    raise ValueError(
        f"Unsupported documentation format: {suffix}. "
        f"Supported: .md, .txt, .json, .docx"
    )


# ====================================================
# LLM HELPERS
# ====================================================


def gpt_client(
    endpoint: str,
    key: str,
    version: str,
    model: str,
    prompt: str,
    system_message: Optional[str] = None,
    temperature: float = 0.0
) -> str:
    try:
        client = create_client(endpoint, key, version)
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=300
        )
        response = completion.choices[0].message.content or ""
        logging.info("gpt_client - Response received successfully.")
        return response
    except Exception as e:
        logging.exception("gpt_client - LLM call failed: %s", e)
        return ""


def summarize_api_documentation(doc_text: str) -> Dict[str, Any]:
    prompt = f"""
You are an API documentation analyst.

From the documentation below, extract a compact, execution-focused API spec.

Return ONLY valid JSON with this schema:
{{
  "api_goal": "string",
  "auth": {{
    "type": "none|bearer|api_key|basic|other",
    "details": "string"
  }},
  "base_url": "string",
  "endpoints": [
    {{
      "path": "string",
      "method": "GET|POST|PUT|PATCH|DELETE|OTHER",
      "purpose": "string",
      "request_fields": ["field1", "field2"],
      "response_fields": ["field1", "field2"],
      "curl_example_present": true
    }}
  ],
  "constraints": ["string"],
  "test_intents": ["string", "string", "string"]
}}

Documentation:
{truncate_text(doc_text, 45000)}
"""
    response = gpt_client(
        endpoint=GENERATOR_ENDPOINT,
        key=GENERATOR_KEY,
        version=GENERATOR_VERSION,
        model=GENERATOR_MODEL,
        prompt=prompt,
        system_message="You extract API specs from raw documentation."
    )

    data = parse_json_safely(response)
    if not isinstance(data, dict):
        raise ValueError("Failed to parse API documentation summary.")
    return data


# ====================================================
# GENERATE QUESTIONS
# ====================================================


def generate_seed_question(topic: str, api_summary: Dict[str, Any]) -> str:
    prompt = f"""
You are generating test cases for an API from its documentation.

API Summary:
{json.dumps(api_summary, indent=2)}

Generate ONE realistic user question for topic: "{topic}"

Requirements:
- Must align with API capabilities and constraints.
- Must be specific and testable.
- Must not be vague.
- Must reflect documentation terminology.

Return ONLY JSON:
{{"prompt":"..."}}
"""
    response = gpt_client(
        endpoint=GENERATOR_ENDPOINT,
        key=GENERATOR_KEY,
        version=GENERATOR_VERSION,
        model=GENERATOR_MODEL,
        prompt=prompt
    )

    data = parse_json_safely(response)
    return data["prompt"]


def generate_followups(seed_prompt: str, api_response: str, api_summary: Dict[str, Any]) -> List[str]:
    prompt = f"""
You are creating test cases for an API based on documentation.

API Summary:
{json.dumps(api_summary, indent=2)}

Original Question:
{seed_prompt}

API Response:
{api_response}

Create exactly {FOLLOWUPS_PER_SEED} follow-up questions.

Requirements:
1. Same functional area as original.
2. Increase complexity progressively.
3. Include functional perspective.
4. Include validation or edge-case perspective.
5. Include operations/error-handling perspective.
6. Keep them distinct and non-repetitive.

Return JSON only:
[
  {{"prompt":"..."}},
  {{"prompt":"..."}},
  {{"prompt":"..."}},
  {{"prompt":"..."}}
]
"""
    response = gpt_client(
        endpoint=GENERATOR_ENDPOINT,
        key=GENERATOR_KEY,
        version=GENERATOR_VERSION,
        model=GENERATOR_MODEL,
        prompt=prompt
    )

    data = parse_json_safely(response)
    if not isinstance(data, list):
        raise ValueError("Follow-up generation did not return a JSON list.")
    return [x["prompt"] for x in data if isinstance(x, dict) and "prompt" in x]


# ====================================================
# API CALL
# ====================================================


def call_target_api(prompt: str) -> str:
    payload = {API_QUERY_FIELD: prompt}

    if API_METHOD.upper() == "GET":
        response = requests.get(
            API_URL,
            headers=API_HEADERS,
            params=payload,
            verify=API_VERIFY_SSL,
            timeout=API_TIMEOUT
        )
    else:
        response = requests.post(
            API_URL,
            headers=API_HEADERS,
            json=payload,
            verify=API_VERIFY_SSL,
            timeout=API_TIMEOUT
        )

    print("\nAPI STATUS:", response.status_code)

    try:
        data = response.json()
        if isinstance(data, dict):
            if "answer" in data:
                return str(data["answer"])
            return json.dumps(data, ensure_ascii=False)
        return str(data)
    except Exception:
        return response.text


# ====================================================
# EXPECTED ANSWER + EVALUATION
# ====================================================


def generate_expected_answer(prompt: str, api_summary: Dict[str, Any]) -> str:
    response = gpt_client(
        endpoint=EXPECTED_ENDPOINT,
        key=EXPECTED_KEY,
        version=EXPECTED_VERSION,
        model=EXPECTED_MODEL,
        prompt=f"""
API Summary:
{json.dumps(api_summary, indent=2)}

User Query:
{prompt}

Generate the expected answer a correctly functioning API/system should return.
Keep it concise and aligned to documented behavior.
""",
        system_message=(
            "You are an API QA expert. "
            "Generate expected outputs from API documentation, concisely."
        )
    )
    return response.strip()


def evaluate_response(expected: str, actual: str, prompt: str) -> str:
    response = gpt_client(
        endpoint=EVAL_ENDPOINT,
        key=EVAL_KEY,
        version=EVAL_VERSION,
        model=EVAL_MODEL,
        prompt=f"""
Compare expected output and actual API output for this query.

Query:
{prompt}

Expected:
{expected}

Actual:
{actual}

Return ONLY one token:
Yes
or
No

Rules:
- Yes if intent and core correctness match.
- Minor wording/detail differences are acceptable.
- No if wrong intent, wrong data meaning, or unrelated response.
"""
    )
    answer = response.strip()
    if answer.lower() not in {"yes", "no"}:
        return "Evaluation Failed"
    return "Yes" if answer.lower() == "yes" else "No"


# ====================================================
# MAIN EXECUTION
# ====================================================


def pick_topics_from_summary(api_summary: Dict[str, Any], fallback: List[str]) -> List[str]:
    topics = []
    for item in api_summary.get("test_intents", []):
        if isinstance(item, str) and item.strip():
            topics.append(item.strip())
    if not topics:
        topics = fallback
    return topics


def run():
    if not API_DOC_PATH:
        raise ValueError("Please set API_DOC_PATH to your documentation file path.")
    if not API_URL:
        raise ValueError("Please set API_URL.")
    if not all([GENERATOR_ENDPOINT, GENERATOR_KEY, GENERATOR_VERSION, GENERATOR_MODEL]):
        raise ValueError("Generator LLM config is incomplete.")
    if not all([EXPECTED_ENDPOINT, EXPECTED_KEY, EXPECTED_VERSION, EXPECTED_MODEL]):
        raise ValueError("Expected-answer LLM config is incomplete.")
    if not all([EVAL_ENDPOINT, EVAL_KEY, EVAL_VERSION, EVAL_MODEL]):
        raise ValueError("Evaluator LLM config is incomplete.")

    print("\nLoading API documentation ...")
    doc_text = load_document_text(API_DOC_PATH)

    print("Summarizing documentation with LLM ...")
    api_summary = summarize_api_documentation(doc_text)

    topics = pick_topics_from_summary(api_summary, FALLBACK_TOPICS)
    print(f"Topics used for seed generation: {topics[:10]}")

    results: List[Dict[str, Any]] = []
    all_prompts: List[str] = []

    print(f"\nGenerating {NUM_SEEDS} seed groups ...")

    for seed_num in range(NUM_SEEDS):
        try:
            print(f"\nSeed Group {seed_num + 1}")
            topic = topics[seed_num % len(topics)]

            seed_question = generate_seed_question(topic, api_summary)
            print("Seed:", seed_question)

            seed_response = call_target_api(seed_question)
            followups = generate_followups(seed_question, seed_response, api_summary)

            current_group = [seed_question] + followups
            all_prompts.extend(current_group)

        except Exception as e:
            print(f"Failed generating seed group {seed_num + 1}: {e}")

    print(f"\nTotal Prompts Generated = {len(all_prompts)}")

    all_prompts = all_prompts[:MAX_TESTS]
    print(f"Final Test Cases Count = {len(all_prompts)}")

    for idx, prompt in enumerate(all_prompts):
        print(f"\nRunning Test Case {idx + 1} / {len(all_prompts)}")

        try:
            expected_output = generate_expected_answer(prompt, api_summary)
        except Exception as e:
            expected_output = f"EXPECTED_ERROR: {str(e)}"

        try:
            api_response = call_target_api(prompt)
        except Exception as e:
            api_response = f"API_ERROR: {str(e)}"

        try:
            worked = evaluate_response(expected_output, api_response, prompt)
        except Exception as e:
            worked = f"EVAL_ERROR: {str(e)}"

        comparison_result = worked if worked in {"Yes", "No", "Evaluation Failed"} else "Evaluation Failed"

        results.append({
            "Test ID": idx + 1,
            "Prompt": prompt,
            "Expected Output": expected_output,
            "API Response": api_response,
            "Comparison Result": comparison_result,
            "Worked As Intended": comparison_result
        })

    total_tests = len(results)
    passed = len([r for r in results if r["Worked As Intended"] == "Yes"])
    failed = len([r for r in results if r["Worked As Intended"] == "No"])
    eval_failed = len([r for r in results if r["Worked As Intended"] == "Evaluation Failed"])

    success_rate = round((passed / total_tests) * 100, 2) if total_tests > 0 else 0.0

    print("\n==========================")
    print("EXECUTION SUMMARY")
    print("==========================")
    print(f"Total Tests      : {total_tests}")
    print(f"Passed           : {passed}")
    print(f"Failed           : {failed}")
    print(f"EvaluationFailed : {eval_failed}")
    print(f"Success Rate     : {success_rate}%")

    df = pd.DataFrame(results)
    summary_df = pd.DataFrame([{
        "Total Tests": total_tests,
        "Passed": passed,
        "Failed": failed,
        "Evaluation Failed": eval_failed,
        "Success Rate (%)": success_rate
    }])

    output_file = "RegIntel_Test_Report_50Cases.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Test Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print("\nSUCCESS")
    print("Excel Generated:")
    print(output_file)


if __name__ == "__main__":
    run()