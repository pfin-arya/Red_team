import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from openai import AzureOpenAI

try:
    from docx import Document  # pip install python-docx
except Exception:
    Document = None

# =========================================================
# 1) USER CONFIG
# =========================================================
API_DOC_PATH = r"C:\Users\CINT037\OneDrive - Poonawalla Fincorp Limited\Desktop\Red_team_1\API Documentation - RegIntel-case1"  # e.g. r"C:\docs\api_spec.docx"
NUM_SEEDS = 1
FOLLOWUPS_PER_SEED = 4
MAX_TESTS = 5
REQUEST_TIMEOUT = 60
VERIFY_SSL = False

# Single Azure endpoint/key/version, only models differ
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_KEY = ""
AZURE_OPENAI_VERSION = ""

# LLM-A: prompt + expected answer generator
MODEL_GEN = ""

# LLM-B: evaluator (expected vs actual)
MODEL_EVAL = ""

OUTPUT_EXCEL = "Auto_API_Test_Report.xlsx"

# =========================================================
# 2) LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================================================
# 3) AZURE OPENAI CLIENT
# =========================================================
_CLIENT: Optional[AzureOpenAI] = None


def get_client() -> AzureOpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_VERSION
        )
    return _CLIENT


def llm_chat(model: str, prompt: str, system: Optional[str] = None, temperature: float = 0.0) -> str:
    try:
        client = get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=300
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logging.exception("LLM call failed: %s", e)
        return ""


# =========================================================
# 4) COMMON HELPERS
# =========================================================
def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json|bash|sh|python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_safely(text: str) -> Any:
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
        if m:
            return json.loads(m.group(1))
        raise


def truncate_text(text: str, limit: int = 45000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n\n[TRUNCATED]"


# =========================================================
# 5) DOCUMENT LOADING (md/txt/json/docx)
# =========================================================
def markdownize_docx(doc: Document) -> str:
    import docx
    lines: List[str] = []
    for element in doc.element.body:
        if element.tag.endswith("p"):
            para = docx.text.paragraph.Paragraph(element, doc)
            if para.text.strip():
                lines.append(para.text.strip())
                lines.append("")
        elif element.tag.endswith("tbl"):
            table = docx.table.Table(element, doc)
            if table.rows:
                headers = [c.text.strip() for c in table.rows[0].cells]
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in table.rows[1:]:
                    cells = [c.text.strip() for c in row.cells]
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")
    return "\n".join(lines).strip()


def load_document_text(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Documentation not found: {path_str}")

    ext = path.suffix.lower()
    if ext in [".md", ".txt", ".log", ".yaml", ".yml", ".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".json":
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    if ext == ".docx":
        if Document is None:
            raise ImportError("Install python-docx: pip install python-docx")
        return markdownize_docx(Document(str(path)))
    raise ValueError(f"Unsupported file type: {ext}. Use .md/.txt/.json/.docx")


# =========================================================
# 6) cURL PARSER + API EXTRACTION FROM DOC
# =========================================================
def parse_curl(curl_cmd: str) -> Dict[str, Any]:
    """
    Parses a cURL command to {method, url, headers, body}.
    Supports common flags: -X, -H, --header, -d, --data, --data-raw.
    """
    tokens = shlex.split(curl_cmd.strip())
    if not tokens or tokens[0].lower() != "curl":
        raise ValueError("Not a valid curl command.")

    method = "GET"
    headers: Dict[str, str] = {}
    data_payload = None
    url = ""

    i = 1
    while i < len(tokens):
        t = tokens[i]
        if t in ["-X", "--request"] and i + 1 < len(tokens):
            method = tokens[i + 1].upper()
            i += 2
            continue
        if t in ["-H", "--header"] and i + 1 < len(tokens):
            h = tokens[i + 1]
            if ":" in h:
                k, v = h.split(":", 1)
                headers[k.strip()] = v.strip()
            i += 2
            continue
        if t in ["-d", "--data", "--data-raw", "--data-binary"] and i + 1 < len(tokens):
            data_payload = tokens[i + 1]
            if method == "GET":
                method = "POST"
            i += 2
            continue
        if t.startswith("http://") or t.startswith("https://"):
            url = t
            i += 1
            continue
        i += 1

    body = None
    if data_payload:
        try:
            body = json.loads(data_payload)
        except Exception:
            body = data_payload

    if not url:
        raise ValueError("URL not found in curl command.")

    return {
        "method": method,
        "url": url,
        "headers": headers,
        "body": body
    }


def extract_api_config_from_doc(doc_text: str) -> Dict[str, Any]:
    """
    Uses LLM-A to extract:
    - api_goal
    - auth hints
    - endpoint method/url
    - sample curl
    - fallback request body template
    """
    prompt = f"""
You are an API documentation parser.

From the documentation below, extract execution-ready API config.

Return ONLY valid JSON:
{{
  "api_goal": "string",
  "auth": {{
    "type": "none|bearer|api_key|basic|other",
    "details": "string"
  }},
  "best_endpoint": {{
    "method": "GET|POST|PUT|PATCH|DELETE",
    "url": "https://...",
    "why": "string"
  }},
  "query_field_candidates": ["Query", "prompt", "question", "input"],
  "request_template": {{}},
  "curl_example": "full curl command or empty string",
  "test_intents": ["string", "string", "string"]
}}

Documentation:
{truncate_text(doc_text)}
"""
    raw = llm_chat(
        model=MODEL_GEN,
        system="Extract accurate API execution details from docs. Do not hallucinate missing values.",
        prompt=prompt
    )
    data = parse_json_safely(raw)
    if not isinstance(data, dict):
        raise ValueError("Could not parse API config from documentation.")
    return data


def build_runtime_api_config(api_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Priority:
    1) Parse curl if present
    2) Else use best_endpoint + request_template
    """
    curl_cmd = (api_meta.get("curl_example") or "").strip()

    if curl_cmd:
        try:
            c = parse_curl(curl_cmd)
            return {
                "method": c["method"],
                "url": c["url"],
                "headers": c["headers"] or {"Content-Type": "application/json"},
                "template_body": c["body"] if c["body"] is not None else {},
                "query_fields": api_meta.get("query_field_candidates") or ["Query", "prompt", "question", "input"]
            }
        except Exception as e:
            logging.warning("cURL parse failed, falling back to endpoint/template: %s", e)

    best = api_meta.get("best_endpoint", {}) if isinstance(api_meta.get("best_endpoint"), dict) else {}
    method = (best.get("method") or "POST").upper()
    url = best.get("url") or ""
    if not url:
        raise ValueError("No endpoint URL found in documentation metadata.")

    headers = {"Content-Type": "application/json"}
    template = api_meta.get("request_template", {})
    return {
        "method": method,
        "url": url,
        "headers": headers,
        "template_body": template if isinstance(template, (dict, list, str)) else {},
        "query_fields": api_meta.get("query_field_candidates") or ["Query", "prompt", "question", "input"]
    }


# =========================================================
# 7) REQUEST BODY BUILDER
# =========================================================
def inject_prompt_into_body(template_body: Any, prompt: str, candidates: List[str]) -> Any:
    """
    Insert test prompt into body template.
    If dict contains a candidate key, replace it.
    Else create {"Query": prompt}.
    """
    if isinstance(template_body, dict):
        body = dict(template_body)
        for k in body.keys():
            if k.lower() in [x.lower() for x in candidates]:
                body[k] = prompt
                return body
        # fallback
        body["Query"] = prompt
        return body
    if isinstance(template_body, str) and template_body.strip():
        # no safe replacement pattern known, fallback as wrapper
        return {"Query": prompt}
    return {"Query": prompt}


# =========================================================
# 8) LLM-A: PROMPT + EXPECTED OUTPUT GENERATION
# =========================================================
def generate_seed_question(topic: str, api_meta: Dict[str, Any]) -> str:
    p = f"""
Generate ONE realistic user test query for this API topic.

Topic: {topic}
API Goal: {api_meta.get("api_goal", "")}
Auth type: {api_meta.get("auth", {}).get("type", "")}
Best endpoint: {api_meta.get("best_endpoint", {})}

Return ONLY JSON:
{{"prompt":"..."}}
"""
    out = llm_chat(MODEL_GEN, p)
    return parse_json_safely(out)["prompt"]


def generate_followups(seed_prompt: str, seed_response: str, api_meta: Dict[str, Any]) -> List[str]:
    p = f"""
Create exactly {FOLLOWUPS_PER_SEED} follow-up test queries.

API Goal:
{api_meta.get("api_goal", "")}

Original Query:
{seed_prompt}

API Response:
{seed_response}

Requirements:
1. Same functional context
2. Increasing complexity
3. Include edge/error-handling angle
4. Distinct wording

Return ONLY JSON:
[
  {{"prompt":"..."}},
  {{"prompt":"..."}},
  {{"prompt":"..."}},
  {{"prompt":"..."}}
]
"""
    out = llm_chat(MODEL_GEN, p)
    data = parse_json_safely(out)
    if not isinstance(data, list):
        raise ValueError("Follow-ups not returned as JSON list.")
    return [x["prompt"] for x in data if isinstance(x, dict) and "prompt" in x]


def generate_expected_output(user_prompt: str, api_meta: Dict[str, Any]) -> str:
    p = f"""
You are generating expected API behavior output.

API Goal:
{api_meta.get("api_goal", "")}

User Query:
{user_prompt}

Return concise expected output aligned with documentation intent.
"""
    return llm_chat(MODEL_GEN, p, system="Generate expected outputs for API QA.").strip()


# =========================================================
# 9) TARGET API CALL
# =========================================================
def call_target_api(user_prompt: str, runtime_cfg: Dict[str, Any]) -> str:
    method = runtime_cfg["method"].upper()
    url = runtime_cfg["url"]
    headers = runtime_cfg["headers"]
    template_body = runtime_cfg["template_body"]
    query_fields = runtime_cfg["query_fields"]

    payload = inject_prompt_into_body(template_body, user_prompt, query_fields)

    if method == "GET":
        response = requests.get(
            url,
            headers=headers,
            params=payload if isinstance(payload, dict) else {"Query": user_prompt},
            timeout=REQUEST_TIMEOUT,
            verify=VERIFY_SSL
        )
    else:
        response = requests.request(
            method,
            url,
            headers=headers,
            json=payload if isinstance(payload, (dict, list)) else {"Query": user_prompt},
            timeout=REQUEST_TIMEOUT,
            verify=VERIFY_SSL
        )

    print(f"\nAPI STATUS: {response.status_code}")
    print("REQUEST URL:", url)
    print("REQUEST METHOD:", method)
    print("REQUEST BODY:", payload)

    try:
        data = response.json()
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return response.text


# =========================================================
# 10) LLM-B: EVALUATION (EXPECTED vs ACTUAL)
# =========================================================
def evaluate_output(expected: str, actual: str, user_prompt: str) -> str:
    p = f"""
Compare expected output and actual API output.

User Query:
{user_prompt}

Expected:
{expected}

Actual:
{actual}

Return ONLY:
Yes
or
No

Yes = same core intent/meaning.
No = wrong intent, mismatch, or unrelated result.
"""
    out = llm_chat(MODEL_EVAL, p).strip().lower()
    if out == "yes":
        return "Yes"
    if out == "no":
        return "No"
    return "Evaluation Failed"


# =========================================================
# 11) TEST ORCHESTRATION
# =========================================================
def pick_topics(api_meta: Dict[str, Any]) -> List[str]:
    intents = api_meta.get("test_intents", [])
    topics = [x.strip() for x in intents if isinstance(x, str) and x.strip()]
    if topics:
        return topics
    return ["General Functional Validation", "Error Handling", "Compliance/Policy Validation"]


def run_test_cycle(api_meta: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    topics = pick_topics(api_meta)
    all_prompts: List[str] = []
    results: List[Dict[str, Any]] = []

    print(f"\nGenerating {NUM_SEEDS} seed groups...")
    for i in range(NUM_SEEDS):
        try:
            topic = topics[i % len(topics)]
            print(f"\nSeed Group {i + 1} | Topic: {topic}")

            seed = generate_seed_question(topic, api_meta)
            seed_resp = call_target_api(seed, runtime_cfg)
            fups = generate_followups(seed, seed_resp, api_meta)

            all_prompts.extend([seed] + fups)
        except Exception as e:
            print(f"Seed group {i + 1} failed: {e}")

    all_prompts = all_prompts[:MAX_TESTS]
    print(f"\nFinal Test Cases Count = {len(all_prompts)}")

    for idx, p in enumerate(all_prompts, start=1):
        print(f"\nRunning Test Case {idx}/{len(all_prompts)}")
        try:
            expected = generate_expected_output(p, api_meta)
        except Exception as e:
            expected = f"EXPECTED_ERROR: {e}"

        try:
            actual = call_target_api(p, runtime_cfg)
        except Exception as e:
            actual = f"API_ERROR: {e}"

        try:
            verdict = evaluate_output(expected, actual, p)
        except Exception as e:
            verdict = f"EVAL_ERROR: {e}"

        worked = verdict if verdict in {"Yes", "No", "Evaluation Failed"} else "Evaluation Failed"

        results.append({
            "Test ID": idx,
            "Prompt": p,
            "Expected Output": expected,
            "API Response": actual,
            "Comparison Result": verdict,
            "Worked As Intended": worked
        })
    return results


# =========================================================
# 12) REPORTING
# =========================================================
def save_report(results: List[Dict[str, Any]], out_file: str) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["Worked As Intended"] == "Yes")
    failed = sum(1 for r in results if r["Worked As Intended"] == "No")
    eval_failed = sum(1 for r in results if r["Worked As Intended"] == "Evaluation Failed")
    success = round((passed / total) * 100, 2) if total else 0.0

    print("\n==========================")
    print("EXECUTION SUMMARY")
    print("==========================")
    print(f"Total Tests      : {total}")
    print(f"Passed           : {passed}")
    print(f"Failed           : {failed}")
    print(f"EvaluationFailed : {eval_failed}")
    print(f"Success Rate     : {success}%")

    df = pd.DataFrame(results)
    summary = pd.DataFrame([{
        "Total Tests": total,
        "Passed": passed,
        "Failed": failed,
        "Evaluation Failed": eval_failed,
        "Success Rate (%)": success
    }])

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Test Results", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)

    print("\nSUCCESS")
    print("Excel Generated:", out_file)


# =========================================================
# 13) ENTRYPOINT
# =========================================================
def validate_config() -> None:
    if not API_DOC_PATH:
        raise ValueError("Set API_DOC_PATH.")
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY or not AZURE_OPENAI_VERSION:
        raise ValueError("Set AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_KEY / AZURE_OPENAI_VERSION.")
    if not MODEL_GEN or not MODEL_EVAL:
        raise ValueError("Set MODEL_GEN and MODEL_EVAL.")


def main() -> None:
    validate_config()

    print("\nLoading API documentation...")
    doc_text = load_document_text(API_DOC_PATH)

    print("Extracting API metadata from documentation...")
    api_meta = extract_api_config_from_doc(doc_text)

    print("Building runtime API configuration...")
    runtime_cfg = build_runtime_api_config(api_meta)

    print("\nResolved API config:")
    print(json.dumps({
        "method": runtime_cfg["method"],
        "url": runtime_cfg["url"],
        "headers": runtime_cfg["headers"],
        "query_fields": runtime_cfg["query_fields"]
    }, indent=2))

    results = run_test_cycle(api_meta, runtime_cfg)
    save_report(results, OUTPUT_EXCEL)


if __name__ == "__main__":
    main()