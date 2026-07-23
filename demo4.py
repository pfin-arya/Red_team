import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Target API config
API_URL = ""
API_METHOD = "POST"  # POST / GET
API_TIMEOUT = 60
API_VERIFY_SSL = False
API_HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer <token>"
}
API_QUERY_FIELD = "Query"

# Documentation input
API_DOC_PATH = "C:\\Users\\CINT037\\OneDrive - Poonawalla Fincorp Limited\\Desktop\\Red_team_1\\pai_chat.md"  # e.g. r"C:\path\to\api_doc.docx"

# Test generation controls
NUM_SEEDS = 10
FOLLOWUPS_PER_SEED = 4
MAX_TESTS = 50

# Optional fallback topics if docs do not contain test intents
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
# SINGLE AZURE OPENAI CONFIG (MINIMIZED VARIABLES)
# =====================================================

AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_KEY = ""
AZURE_OPENAI_VERSION = "2025-01-01-preview"

# Only model names differ
MODEL_GENERATOR = ""
MODEL_EXPECTED = ""
MODEL_EVALUATOR = ""

# ====================================================
# LOGGING
# ====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ====================================================
# CLIENT
# ====================================================


def create_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION
    )


CLIENT = None


def get_client() -> AzureOpenAI:
    global CLIENT
    if CLIENT is None:
        CLIENT = create_client()
    return CLIENT


# ====================================================
# UTILS
# ====================================================


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json|python|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_safely(text: str) -> Any:
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
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
    text_parts: List[str] = []
    import docx

    for element in doc.element.body:
        if element.tag.endswith("p"):
            para = docx.text.paragraph.Paragraph(element, doc)
            if para.text.strip():
                text_parts.append(para.text.strip())
                text_parts.append("")
        elif element.tag.endswith("tbl"):
            table = docx.table.Table(element, doc)
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
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)

    if suffix == ".docx":
        if Document is None:
            raise ImportError("Install python-docx for .docx support: pip install python-docx")
        return markdownize_docx(Document(file_path))

    raise ValueError(
        f"Unsupported format: {suffix}. Supported: .md, .txt, .json, .docx"
    )


# ====================================================
# LLM CALL
# ====================================================


def llm_chat(
    model: str,
    prompt: str,
    system_message: Optional[str] = None,
    temperature: float = 0.0
) -> str:
    try:
        client = get_client()
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
        return completion.choices[0].message.content or ""
    except Exception as e:
        logging.exception("LLM call failed: %s", e)
        return ""


# ====================================================
# DOC SUMMARY
# ====================================================


def summarize_api_documentation(doc_text: str) -> Dict[str, Any]:
    prompt = f"""
You are an API documentation analyst.

From the documentation below, extract a compact API spec.

Return ONLY valid JSON:
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
    response = llm_chat(
        model=MODEL_GENERATOR,
        prompt=prompt,
        system_message="You extract API specs from raw documentation."
    )
    data = parse_json_safely(response)
    if not isinstance(data, dict):
        raise ValueError("Unable to parse documentation summary.")
    return data


# ====================================================
# TEST PROMPT GENERATION
# ====================================================


def generate_seed_question(topic: str, api_summary: Dict[str, Any]) -> str:
    prompt = f"""
You are generating test cases for an API.

API Summary:
{json.dumps(api_summary, indent=2)}

Generate ONE realistic user question for topic: "{topic}"

Requirements:
- Must align with API capabilities and constraints.
- Must be specific and testable.
- Must use doc terminology.

Return ONLY JSON:
{{"prompt":"..."}}
"""
    response = llm_chat(model=MODEL_GENERATOR, prompt=prompt)
    return parse_json_safely(response)["prompt"]


def generate_followups(seed_prompt: str, api_response: str, api_summary: Dict[str, Any]) -> List[str]:
    prompt = f"""
You are creating follow-up test cases for an API.

API Summary:
{json.dumps(api_summary, indent=2)}

Original Question:
{seed_prompt}

API Response:
{api_response}

Create exactly {FOLLOWUPS_PER_SEED} follow-up questions.

Requirements:
1. Same functional area.
2. Progressive complexity.
3. Include edge/error perspective.
4. Distinct wording.

Return JSON only:
[
  {{"prompt":"..."}},
  {{"prompt":"..."}},
  {{"prompt":"..."}},
  {{"prompt":"..."}}
]
"""
    response = llm_chat(model=MODEL_GENERATOR, prompt=prompt)
    data = parse_json_safely(response)
    if not isinstance(data, list):
        raise ValueError("Follow-up generation did not return JSON list.")
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
    return llm_chat(
        model=MODEL_EXPECTED,
        system_message="You are an API QA expert generating expected outputs.",
        prompt=f"""
API Summary:
{json.dumps(api_summary, indent=2)}

User Query:
{prompt}

Generate concise expected output aligned with documented behavior.
"""
    ).strip()


def evaluate_response(expected: str, actual: str, prompt: str) -> str:
    resp = llm_chat(
        model=MODEL_EVALUATOR,
        prompt=f"""
Compare expected and actual output for the query.

Query:
{prompt}

Expected:
{expected}

Actual:
{actual}

Return ONLY:
Yes
or
No

Rules:
- Yes if core intent/meaning matches.
- Minor wording differences are acceptable.
- No if intent/data meaning is wrong or unrelated.
"""
    ).strip().lower()

    if resp == "yes":
        return "Yes"
    if resp == "no":
        return "No"
    return "Evaluation Failed"


# ====================================================
# MAIN
# ====================================================


def pick_topics_from_summary(api_summary: Dict[str, Any]) -> List[str]:
    intents = api_summary.get("test_intents", [])
    topics = [x.strip() for x in intents if isinstance(x, str) and x.strip()]
    return topics if topics else FALLBACK_TOPICS


def validate_config() -> None:
    if not API_DOC_PATH:
        raise ValueError("Set API_DOC_PATH.")
    if not API_URL:
        raise ValueError("Set API_URL.")
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY or not AZURE_OPENAI_VERSION:
        raise ValueError("Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_VERSION.")
    if not MODEL_GENERATOR or not MODEL_EXPECTED or not MODEL_EVALUATOR:
        raise ValueError("Set MODEL_GENERATOR, MODEL_EXPECTED, MODEL_EVALUATOR.")


def run() -> None:
    validate_config()

    print("\nLoading API documentation...")
    doc_text = load_document_text(API_DOC_PATH)

    print("Summarizing API documentation...")
    api_summary = summarize_api_documentation(doc_text)

    topics = pick_topics_from_summary(api_summary)
    print(f"Topics selected: {topics[:10]}")

    all_prompts: List[str] = []
    results: List[Dict[str, Any]] = []

    print(f"\nGenerating {NUM_SEEDS} seed groups...")

    for seed_num in range(NUM_SEEDS):
        try:
            print(f"\nSeed Group {seed_num + 1}")
            topic = topics[seed_num % len(topics)]

            seed_question = generate_seed_question(topic, api_summary)
            print("Seed:", seed_question)

            seed_response = call_target_api(seed_question)
            followups = generate_followups(seed_question, seed_response, api_summary)

            all_prompts.extend([seed_question] + followups)

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
            expected_output = f"EXPECTED_ERROR: {e}"

        try:
            api_response = call_target_api(prompt)
        except Exception as e:
            api_response = f"API_ERROR: {e}"

        try:
            verdict = evaluate_response(expected_output, api_response, prompt)
        except Exception as e:
            verdict = f"EVAL_ERROR: {e}"

        worked = verdict if verdict in {"Yes", "No", "Evaluation Failed"} else "Evaluation Failed"

        results.append({
            "Test ID": idx + 1,
            "Prompt": prompt,
            "Expected Output": expected_output,
            "API Response": api_response,
            "Comparison Result": verdict,
            "Worked As Intended": worked
        })

    total_tests = len(results)
    passed = sum(1 for r in results if r["Worked As Intended"] == "Yes")
    failed = sum(1 for r in results if r["Worked As Intended"] == "No")
    eval_failed = sum(1 for r in results if r["Worked As Intended"] == "Evaluation Failed")
    success_rate = round((passed / total_tests) * 100, 2) if total_tests else 0.0

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