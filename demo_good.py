import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import pandas as pd
import requests
from openai import AzureOpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from docx import Document  # pip install python-docx
except Exception:
    Document = None

# =========================================================
# 1) USER CONFIG
# =========================================================
API_DOC_PATH = r"C:\Users\CINT037\OneDrive - Poonawalla Fincorp Limited\Desktop\Red_team_1\VOC_Categorisation_API_Documentation"

NUM_SEEDS = 10
FOLLOWUPS_PER_SEED = 4
MAX_TESTS = 50
REQUEST_TIMEOUT = 60
VERIFY_SSL = False



# Default employee email to use in test queries if not provided in docs or prompts
#DEFAULT_EMPLOYEE_EMAIL = "kriti.khare@poonawallafincorp.com"

# Strict mode to avoid schema mismatch on sensitive endpoints
STRICT_REQUEST_FROM_TEMPLATE = True

# Add hard auth headers if required by endpoint
FORCE_AUTH_HEADERS: Dict[str, str] = {
    # "Authorization": "Bearer <token>",
    # "x-functions-key": "<function-key>"
}

PRINT_RESPONSE_TEXT_ON_ERROR = True

# =========================================================
# 2) LOGGING + HTTP RETRY SESSION
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

SESSION = requests.Session()
RETRY = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
)
SESSION.mount("https://", HTTPAdapter(max_retries=RETRY))
SESSION.mount("http://", HTTPAdapter(max_retries=RETRY))

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


def llm_chat(model: str, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
    try:
        client = get_client()
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": 300
        }

        if temperature is not None:
            params["temperature"] = temperature

        resp = client.chat.completions.create(**params)
        return (resp.choices[0].message.content or "").strip()

    except Exception as e:
        logging.exception("LLM call failed: %s", e)
        return ""


# =========================================================
# 4) COMMON HELPERS
# =========================================================
def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json|bash|sh|python|text)?\s*", "", text, flags=re.IGNORECASE)
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
        raise ValueError(f"Unable to parse JSON from model output: {cleaned[:500]}")


def truncate_text(text: str, limit: int = 45000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n\n[TRUNCATED]"


def mask_secret_headers(headers: Dict[str, str]) -> Dict[str, str]:
    masked = {}
    for k, v in headers.items():
        lk = k.lower()
        if "authorization" in lk or "key" in lk or "token" in lk or "sig" in lk:
            masked[k] = "***"
        else:
            masked[k] = v
    return masked


def sanitize_url(raw_url: str) -> str:
    url = (raw_url or "").strip().strip('"').strip("'")
    parts = urlsplit(url)
    qs = parse_qsl(parts.query, keep_blank_values=True)
    clean_qs = []
    for k, v in qs:
        vv = v.strip().strip('"').strip("'")
        vv = vv.replace("%22", "").replace("%27", "")
        clean_qs.append((k, vv))
    clean_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(clean_qs), parts.fragment))
    clean_url = clean_url.replace("%22", "").replace("%27", "")
    return clean_url

def extract_vocc_urls_from_text(doc_text: str) -> List[str]:
    pattern = r"https?://[^\s'\"`]+/api/vocc\b"
    found = re.findall(pattern, doc_text or "", flags=re.IGNORECASE)
    cleaned = [sanitize_url(u) for u in found]
    # preserve order, remove duplicates
    seen = set()
    out: List[str] = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def choose_vocc_url(api_meta: Dict[str, Any], doc_text: str) -> str:
    candidates: List[str] = []

    curl_example = (api_meta.get("curl_example") or "").strip()
    if curl_example:
        try:
            c = parse_curl(curl_example)
            cu = sanitize_url(c.get("url") or "")
            if "/api/vocc" in cu.lower():
                candidates.append(cu)
        except Exception:
            pass

    best = api_meta.get("best_endpoint", {}) if isinstance(api_meta.get("best_endpoint"), dict) else {}
    bu = sanitize_url(best.get("url") or "")
    if bu and "/api/vocc" in bu.lower():
        candidates.append(bu)

    candidates.extend(extract_vocc_urls_from_text(doc_text))

    # absolute urls only
    candidates = [u for u in candidates if u.startswith("http://") or u.startswith("https://")]
    if not candidates:
        raise ValueError("No absolute /api/vocc endpoint found in documentation.")

    # prefer non-hr-transformation if multiple candidates exist (doc ambiguity resolver)
    for u in candidates:
        if "hr-transformation" not in u.lower():
            return u
    return candidates[0]

# =========================================================
# 5) DOCUMENT LOADING
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


def resolve_doc_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.exists():
        return p

    if p.suffix == "":
        for ext in [".md", ".txt", ".json", ".docx", ".doc", ".pdf"]:
            c = Path(str(p) + ext)
            if c.exists():
                return c

    parent = p.parent if p.parent.exists() else Path(".")
    nearby = [x.name for x in parent.glob("*")]
    raise FileNotFoundError(
        f"Documentation not found: {p}\n"
        f"Nearby files in {parent}:\n- " + "\n- ".join(nearby[:30])
    )


def load_document_text(path_str: str) -> str:
    path = resolve_doc_path(path_str)
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

    if ext == ".doc":
        raise ValueError("Legacy .doc is not supported directly. Convert to .docx/.md/.txt first.")

    if ext == ".pdf":
        raise ValueError("PDF parsing not implemented in this script. Convert to .md/.txt/.docx first.")

    raise ValueError(f"Unsupported file type: {ext}. Use .md/.txt/.json/.docx")


# =========================================================
# 6) cURL PARSER + API EXTRACTION
# =========================================================
def parse_curl(curl_cmd: str) -> Dict[str, Any]:
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

    body: Any = None
    if data_payload:
        try:
            body = json.loads(data_payload)
        except Exception:
            body = data_payload

    if not url:
        raise ValueError("URL not found in curl command.")

    return {"method": method, "url": url, "headers": headers, "body": body}


def normalize_url(url: str, base_url: str) -> str:
    if not url:
        return url
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if not base_url:
        return url
    return base_url.rstrip("/") + "/" + url.lstrip("/")


def extract_api_config_from_doc(doc_text: str) -> Dict[str, Any]:
    prompt = f"""
You are an API documentation parser.

From the documentation below, extract execution-ready API config.

Return ONLY valid JSON:
{{
  "api_goal": "string",
  "auth": {{
    "type": "none|bearer|api_key|basic|function_key|other",
    "details": "string"
  }},
  "best_endpoint": {{
    "method": "GET|POST|PUT|PATCH|DELETE",
    "url": "absolute or relative url/path",
    "why": "string"
  }},
  "query_field_candidates": ["message", "query", "question", "input", "Query"],
  "request_template": {{}},
  "curl_example": "full curl command or empty string",
  "required_fields": ["employee_email", "message"],
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


def build_runtime_api_config(api_meta: Dict[str, Any], doc_text: str) -> Dict[str, Any]:
    curl_cmd = (api_meta.get("curl_example") or "").strip()

    if curl_cmd:
        try:
            c = parse_curl(curl_cmd)
            cfg = {
                "method": c["method"],
                "url": c["url"],
                "headers": c["headers"] or {"Content-Type": "application/json"},
                "template_body": c["body"] if c["body"] is not None else {},
                "query_fields": api_meta.get("query_field_candidates") or ["message", "query", "question", "input", "Query"],
                "required_fields": api_meta.get("required_fields") or []
            }
        except Exception as e:
            logging.warning("cURL parse failed, fallback to endpoint/template: %s", e)
            cfg = {}
    else:
        cfg = {}

    if not cfg:
        best = api_meta.get("best_endpoint", {}) if isinstance(api_meta.get("best_endpoint"), dict) else {}
        cfg = {
            "method": (best.get("method") or "POST").upper(),
            "url": best.get("url") or "",
            "headers": {"Content-Type": "application/json"},
            "template_body": api_meta.get("request_template", {}) if isinstance(api_meta.get("request_template"), (dict, list, str)) else {},
            "query_fields": api_meta.get("query_field_candidates") or ["message", "query", "question", "input", "Query"],
            "required_fields": api_meta.get("required_fields") or []
        }

    resolved_vocc_url = choose_vocc_url(api_meta, doc_text)
    cfg["url"] = sanitize_url(normalize_url(resolved_vocc_url, API_BASE_URL_OVERRIDE))

    if FUNCTION_KEY_OVERRIDE:
        headers = dict(cfg["headers"])
        headers.setdefault("x-functions-key", FUNCTION_KEY_OVERRIDE)
        cfg["headers"] = headers

    if not cfg["url"] or (not cfg["url"].startswith("http://") and not cfg["url"].startswith("https://")):
        raise ValueError(
            "Resolved endpoint is not absolute URL. "
            "Set API_BASE_URL_OVERRIDE or provide absolute URL in documentation/cURL."
        )

    return cfg


# =========================================================
# 7) REQUEST BODY BUILDER
# =========================================================
def inject_prompt_into_body(template_body: Any, prompt: str, candidates: List[str], required_fields: List[str]) -> Any:
    """
    Only inject prompt into explicit query-like fields.
    Never overwrite contract fields such as path/file/url/blobPath.
    """
    if isinstance(template_body, dict) and template_body:
        body = dict(template_body)
        lower_map = {k.lower(): k for k in body.keys()}

        # Only these keys are safe for NL prompt injection
        safe_prompt_keys = {
            "query", "question", "message", "input", "prompt", "text"
        }

        # 1) candidate keys from docs, but only if query-like
        for c in candidates:
            ck = str(c).lower()
            if ck in safe_prompt_keys and ck in lower_map:
                body[lower_map[ck]] = prompt
                return body

        # 2) required fields from docs, but only if query-like
        for rf in required_fields:
            rk = str(rf).lower()
            if rk in safe_prompt_keys and rk in lower_map:
                body[lower_map[rk]] = prompt
                return body

        # 3) if no query-like field exists, keep template untouched
        return body

    # If no template exists, create a generic query payload
    key = "Query"
    return {key: prompt}


# =========================================================
# 8) LLM-A: PROMPT + EXPECTED OUTPUT
# =========================================================
def generate_seed_question(topic: str, api_meta: Dict[str, Any]) -> str:
    p = f"""
Generate ONE realistic user test query for this API topic.

Topic: {topic}
API Goal: {api_meta.get("api_goal", "")}

Return ONLY JSON:
{{"prompt":"..."}}
"""
    out = llm_chat(MODEL_GEN, p, temperature=0.0)
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
1. Same context
2. Increasing complexity
3. Include edge/error angle
4. Distinct wording

Return ONLY JSON:
[
  {{"prompt":"..."}},
  {{"prompt":"..."}},
  {{"prompt":"..."}},
  {{"prompt":"..."}}
]
"""
    out = llm_chat(MODEL_GEN, p, temperature=0.0)
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
    return llm_chat(MODEL_GEN, p, system="Generate expected outputs for API QA.", temperature=0.0).strip()


# =========================================================
# 9) TARGET API CALL
# =========================================================
def call_target_api(user_prompt: str, runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    method = runtime_cfg["method"].upper()
    url = runtime_cfg["url"]
    headers = dict(runtime_cfg["headers"])
    headers.update(FORCE_AUTH_HEADERS)

    template_body = runtime_cfg["template_body"]
    query_fields = runtime_cfg["query_fields"]
    required_fields = runtime_cfg["required_fields"]

    payload = inject_prompt_into_body(template_body, user_prompt, query_fields, required_fields)

    try:
        if method == "GET":
            response = SESSION.get(
                url,
                headers=headers,
                params=payload if isinstance(payload, dict) else {"Query": user_prompt},
                timeout=REQUEST_TIMEOUT,
                verify=VERIFY_SSL
            )
        else:
            response = SESSION.request(
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
        print("REQUEST HEADERS:", mask_secret_headers(headers))
        print("REQUEST BODY:", payload)

        try:
            parsed = response.json()
            api_text = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            api_text = response.text

        if PRINT_RESPONSE_TEXT_ON_ERROR and response.status_code >= 400:
            print("RESPONSE TEXT:", api_text)

        return {
            "status_code": response.status_code,
            "request_body": payload,
            "response_text": api_text
        }

    except requests.exceptions.RequestException as e:
        return {
            "status_code": -1,
            "request_body": payload,
            "response_text": f"API_ERROR: {e}"
        }

def smoke_test_vocc_from_runtime_cfg(runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single deterministic call using the exact runtime cfg.
    Use this to validate auth/body contract before running LLM-generated cycles.
    """
    method = runtime_cfg["method"].upper()
    url = runtime_cfg["url"]
    headers = dict(runtime_cfg.get("headers", {}))
    headers.update(FORCE_AUTH_HEADERS)
    payload = runtime_cfg.get("template_body", {})

    if not isinstance(payload, (dict, list)):
        payload = {}

    if method != "POST":
        method = "POST"

    try:
        resp = SESSION.request(
            method,
            url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
            verify=VERIFY_SSL
        )

        print("\nSMOKE TEST")
        print("STATUS:", resp.status_code)
        print("URL:", url)
        print("METHOD:", method)
        print("HEADERS:", mask_secret_headers(headers))
        print("BODY:", payload)

        try:
            body_text = json.dumps(resp.json(), ensure_ascii=False)
        except Exception:
            body_text = resp.text

        print("RESPONSE:", body_text)

        return {
            "status_code": resp.status_code,
            "response_text": body_text
        }
    except requests.exceptions.RequestException as e:
        return {
            "status_code": -1,
            "response_text": f"API_ERROR: {e}"
        }


# =========================================================
# 10) LLM-B: EVALUATION
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
# 11) ORCHESTRATION
# =========================================================
def pick_topics(api_meta: Dict[str, Any]) -> List[str]:
    intents = api_meta.get("test_intents", [])
    topics = [x.strip() for x in intents if isinstance(x, str) and x.strip()]
    if topics:
        return topics
    return ["General Functional Validation", "Error Handling", "Policy Validation"]


def run_test_cycle(api_meta: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    topics = pick_topics(api_meta)
    all_prompts: List[str] = []
    results: List[Dict[str, Any]] = []
    seed_errors: List[str] = []
    seed_success = 0

    print(f"\nGenerating {NUM_SEEDS} seed groups...")
    for i in range(NUM_SEEDS):
        try:
            topic = topics[i % len(topics)]
            print(f"\nSeed Group {i + 1} | Topic: {topic}")

            seed = generate_seed_question(topic, api_meta)
            seed_call = call_target_api(seed, runtime_cfg)

            if seed_call["status_code"] == 401:
                raise PermissionError("401 Unauthorized during seed call. Check auth header/signature/body contract.")

            try:
                fups = generate_followups(seed, seed_call["response_text"], api_meta) if FOLLOWUPS_PER_SEED > 0 else []
            except Exception:
                fups = []

            all_prompts.extend([seed] + fups)
            seed_success += 1
        except Exception as e:
            msg = f"Seed group {i + 1} failed: {e}"
            print(msg)
            seed_errors.append(msg)

    all_prompts = all_prompts[:MAX_TESTS]
    print(f"\nFinal Test Cases Count = {len(all_prompts)}")

    if not all_prompts:
        raise RuntimeError(
            "No prompts generated. Check LLM extraction/generation and API auth/config.\n"
            + "\n".join(seed_errors[:5])
        )

    for idx, p in enumerate(all_prompts, start=1):
        print(f"\nRunning Test Case {idx}/{len(all_prompts)}")
        try:
            expected = generate_expected_output(p, api_meta)
        except Exception as e:
            expected = f"EXPECTED_ERROR: {e}"

        try:
            api_call = call_target_api(p, runtime_cfg)
            actual = api_call["response_text"]
            status_code = api_call["status_code"]
            request_body = api_call["request_body"]
        except Exception as e:
            actual = f"API_ERROR: {e}"
            status_code = -1
            request_body = {}

        try:
            verdict = evaluate_output(expected, actual, p) if status_code == 200 else "No"
        except Exception as e:
            verdict = f"EVAL_ERROR: {e}"

        worked = verdict if verdict in {"Yes", "No", "Evaluation Failed"} else "Evaluation Failed"

        results.append({
            "Test ID": idx,
            "Prompt": p,
            "Expected Output": expected,
            "API Response": actual,
            "Request Body": json.dumps(request_body, ensure_ascii=False),
            "Comparison Result": verdict,
            "Worked As Intended": worked
        })

    diagnostics = {
        "seed_success_count": seed_success,
        "total_generated_prompts": len(all_prompts),
        "resolved_url": runtime_cfg["url"],
        "resolved_method": runtime_cfg["method"]
    }
    return results, diagnostics


# =========================================================
# 12) REPORTING
# =========================================================
def save_report(results: List[Dict[str, Any]], diagnostics: Dict[str, Any], out_file: str) -> None:
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
    diag = pd.DataFrame([diagnostics])

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Test Results", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        diag.to_excel(writer, sheet_name="Diagnostics", index=False)

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
    runtime_cfg = build_runtime_api_config(api_meta, doc_text)

    smoke = smoke_test_vocc_from_runtime_cfg(runtime_cfg)
    if smoke["status_code"] == 401:
        raise RuntimeError(
            "401 Unauthorized on deterministic smoke test. "
            "Endpoint/body are now stable; issue is auth/key scope or function auth level."
        )

    print("\nResolved API config:")
    print(json.dumps({
        "method": runtime_cfg["method"],
        "url": runtime_cfg["url"],
        "headers": mask_secret_headers(runtime_cfg["headers"]),
        "query_fields": runtime_cfg["query_fields"],
        "required_fields": runtime_cfg["required_fields"]
    }, indent=2))

    results, diagnostics = run_test_cycle(api_meta, runtime_cfg)
    save_report(results, diagnostics, OUTPUT_EXCEL)


if __name__ == "__main__":
    main()