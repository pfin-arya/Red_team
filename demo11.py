import json
import logging
import re
import shlex
import os
from dotenv import load_dotenv
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

# Optional PDF support (recommended for SOP folders with PDFs)
try:
    from pypdf import PdfReader  # pip install pypdf
except Exception:
    PdfReader = None


# =========================================================
# 1) USER CONFIG
# =========================================================
load_dotenv(".env.local")

# API documentation is mandatory; do not hardcode project-specific paths.
API_DOC_PATH = os.getenv("API_DOC_PATH", "").strip()

# SOP is now mandatory for this experiment, because evaluator must judge against SOP.
SOP_FOLDER_PATH = os.getenv("SOP_FOLDER_PATH", "").strip()
MAX_SOP_FILES = int(os.getenv("MAX_SOP_FILES", "40"))
MAX_SOP_CHARS = int(os.getenv("MAX_SOP_CHARS", "120000"))

NUM_SEEDS = int(os.getenv("NUM_SEEDS", "10"))
FOLLOWUPS_PER_SEED = int(os.getenv("FOLLOWUPS_PER_SEED", "4"))
MAX_TESTS = int(os.getenv("MAX_TESTS", "50"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() in {"1", "true", "yes"}

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "").strip()
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2024-05-01-preview").strip()

MODEL_GEN = os.getenv("MODEL_GEN", "gpt-5.4").strip()
MODEL_EVAL = os.getenv("MODEL_EVAL", "gpt-5.5").strip()

OUTPUT_EXCEL = os.getenv("OUTPUT_EXCEL", "Auto_API_Test_Report_Experimental.xlsx").strip()
API_BASE_URL_OVERRIDE = os.getenv("API_BASE_URL_OVERRIDE", "").strip()
FUNCTION_KEY_OVERRIDE = os.getenv("FUNCTION_KEY_OVERRIDE", "").strip()
DEFAULT_EMPLOYEE_EMAIL = os.getenv("DEFAULT_EMPLOYEE_EMAIL", "employee@example.com").strip()

# Optional hard auth headers as JSON string in .env.local
# Example:
# FORCE_AUTH_HEADERS_JSON={"Authorization":"Bearer <token>","x-functions-key":"<key>"}
try:
    FORCE_AUTH_HEADERS = json.loads(os.getenv("FORCE_AUTH_HEADERS_JSON", "{}"))
    if not isinstance(FORCE_AUTH_HEADERS, dict):
        FORCE_AUTH_HEADERS = {}
except Exception:
    FORCE_AUTH_HEADERS = {}

PRINT_RESPONSE_TEXT_ON_ERROR = os.getenv(
    "PRINT_RESPONSE_TEXT_ON_ERROR", "true"
).lower() in {"1", "true", "yes"}


# =========================================================
# 2) LOGGING + HTTP RETRY SESSION
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

SESSION = requests.Session()
RETRY = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
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
            api_version=AZURE_OPENAI_VERSION,
        )
    return _CLIENT


def llm_chat(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    try:
        client = get_client()
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": 300,
        }
        if temperature is not None:
            params["temperature"] = temperature

        try:
            resp = client.chat.completions.create(**params)
        except Exception as e:
            err_text = str(e).lower()
            if "temperature" in err_text and "unsupported" in err_text and "temperature" in params:
                params.pop("temperature", None)
                resp = client.chat.completions.create(**params)
            else:
                raise

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
    clean_url = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(clean_qs), parts.fragment)
    )
    clean_url = clean_url.replace("%22", "").replace("%27", "")
    return clean_url


def enforce_employee_email(payload: Any, forced_email: str) -> Any:
    if not isinstance(payload, dict):
        return payload
    body = dict(payload)
    for k in ["employee_email", "email", "userEmail"]:
        if k in body:
            body[k] = forced_email
            return body
    body["employee_email"] = forced_email
    return body


def extract_message_text(payload_text: str) -> str:
    """
    Extract semantic message text from expected/actual JSON-like payload.
    Falls back to raw text if JSON parsing fails.
    """
    if not payload_text:
        return ""
    try:
        obj = parse_json_safely(payload_text)
    except Exception:
        return str(payload_text).strip()

    if isinstance(obj, dict):
        msg = obj.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        if isinstance(msg, str) and msg.strip():
            return msg.strip()

        data = obj.get("data")
        if isinstance(data, dict):
            r = data.get("response")
            if isinstance(r, str) and r.strip():
                return r.strip()
            m2 = data.get("message")
            if isinstance(m2, str) and m2.strip():
                return m2.strip()

    return str(payload_text).strip()


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


def read_pdf_text(path: Path) -> str:
    if PdfReader is None:
        raise ImportError("Install pypdf: pip install pypdf")
    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


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
        return read_pdf_text(path)

    raise ValueError(f"Unsupported file type: {ext}. Use .md/.txt/.json/.docx/.pdf")


def load_sop_corpus(folder_path: str, max_files: int, max_chars: int) -> Tuple[str, Dict[str, Any]]:
    """
    Loads SOP files from a folder and returns combined text + diagnostics.
    """
    if not folder_path:
        return "", {
            "sop_provided": False,
            "sop_loaded_files": 0,
            "sop_skipped_files": 0,
            "sop_total_chars": 0,
            "sop_note": "SOP not provided",
            "sop_sources": [],
        }

    p = Path(folder_path).expanduser()
    if not p.exists() or not p.is_dir():
        return "", {
            "sop_provided": False,
            "sop_loaded_files": 0,
            "sop_skipped_files": 0,
            "sop_total_chars": 0,
            "sop_note": f"SOP folder missing/invalid: {p}",
            "sop_sources": [],
        }

    exts = {".pdf", ".docx", ".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".log"}
    files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts]
    files = sorted(files)[:max_files]

    chunks: List[str] = []
    loaded = 0
    skipped = 0
    source_names: List[str] = []

    for f in files:
        try:
            text = load_document_text(str(f))
            if text.strip():
                chunks.append(f"\n\n### SOP_SOURCE: {f.name}\n{text.strip()}")
                loaded += 1
                source_names.append(f.name)
        except Exception:
            skipped += 1

    combined = "\n".join(chunks).strip()
    combined = truncate_text(combined, max_chars)

    provided = loaded > 0
    return combined, {
        "sop_provided": provided,
        "sop_loaded_files": loaded,
        "sop_skipped_files": skipped,
        "sop_total_chars": len(combined),
        "sop_note": "SOP corpus loaded" if provided else "SOP provided but no readable files",
        "sop_sources": source_names,
    }


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
        prompt=prompt,
    )
    data = parse_json_safely(raw)
    if not isinstance(data, dict):
        raise ValueError("Could not parse API config from documentation.")
    return data


def build_runtime_api_config(api_meta: Dict[str, Any]) -> Dict[str, Any]:
    curl_cmd = (api_meta.get("curl_example") or "").strip()

    if curl_cmd:
        try:
            c = parse_curl(curl_cmd)
            cfg = {
                "method": c["method"],
                "url": c["url"],
                "headers": c["headers"] or {"Content-Type": "application/json"},
                "template_body": c["body"] if c["body"] is not None else {},
                "query_fields": api_meta.get("query_field_candidates")
                or ["message", "query", "question", "input", "Query"],
                "required_fields": api_meta.get("required_fields") or [],
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
            "template_body": api_meta.get("request_template", {})
            if isinstance(api_meta.get("request_template"), (dict, list, str))
            else {},
            "query_fields": api_meta.get("query_field_candidates")
            or ["message", "query", "question", "input", "Query"],
            "required_fields": api_meta.get("required_fields") or [],
        }

    cfg["url"] = sanitize_url(normalize_url(cfg["url"], API_BASE_URL_OVERRIDE))

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
def inject_prompt_into_body(
    template_body: Any, prompt: str, candidates: List[str], required_fields: List[str]
) -> Any:
    if isinstance(template_body, dict) and template_body:
        body = dict(template_body)
        lower_map = {k.lower(): k for k in body.keys()}

        for c in candidates:
            key = lower_map.get(c.lower())
            if key:
                body[key] = prompt
                return body

        for rf in required_fields:
            key = lower_map.get(str(rf).lower())
            if key:
                body[key] = prompt
                return body

        first_key = next(iter(body.keys()))
        body[first_key] = prompt
        return body

    key = required_fields[0] if required_fields else (candidates[0] if candidates else "Query")
    return {key: prompt}


# =========================================================
# 8) LLM-A: PROMPT GENERATION ONLY (NO EXPECTED OUTPUT)
# =========================================================
def generate_seed_question(
    topic: str,
    api_meta: Dict[str, Any],
    doc_text: str,
    sop_text: str,
) -> str:
    p = f"""
Generate ONE realistic user test query grounded in SOP + API docs.

Topic:
{topic}

API Goal:
{api_meta.get("api_goal", "")}

Rules:
1) Query must be valid and in-scope.
2) Include necessary details when possible.
3) Keep wording natural and user-like.
4) Do not ask out-of-domain requests.

API Documentation:
{truncate_text(doc_text, 12000)}

SOP Corpus:
{truncate_text(sop_text, 12000)}

Return ONLY valid JSON:
{{
  "prompt": "...",
  "source": "SOP:<document/process> + API_DOC"
}}
"""
    out = llm_chat(MODEL_GEN, p, temperature=0.0)
    data = parse_json_safely(out)
    if not isinstance(data, dict) or "prompt" not in data:
        raise ValueError("Seed question generation failed: invalid JSON output.")
    return str(data["prompt"]).strip()


def generate_followups(
    seed_prompt: str,
    seed_response: str,
    api_meta: Dict[str, Any],
    doc_text: str,
    sop_text: str,
) -> List[str]:
    p = f"""
Create exactly {FOLLOWUPS_PER_SEED} follow-up test queries grounded in SOP + API docs.

Requirements:
1) Keep same conversation context.
2) Increase complexity gradually.
3) Include policy/process edge case coverage.
4) Keep all prompts in API scope.

API Goal:
{api_meta.get("api_goal", "")}

Original Query:
{seed_prompt}

API Response:
{seed_response}

API Documentation:
{truncate_text(doc_text, 10000)}

SOP Corpus:
{truncate_text(sop_text, 10000)}

Return ONLY valid JSON:
[
  {{"prompt":"...","source":"SOP:<rule/doc> + API_DOC"}},
  {{"prompt":"...","source":"SOP:<rule/doc> + API_DOC"}},
  {{"prompt":"...","source":"SOP:<rule/doc> + API_DOC"}},
  {{"prompt":"...","source":"SOP:<rule/doc> + API_DOC"}}
]
"""
    out = llm_chat(MODEL_GEN, p, temperature=0.0)
    data = parse_json_safely(out)
    if not isinstance(data, list):
        raise ValueError("Follow-ups not returned as JSON list.")
    prompts: List[str] = []
    for x in data:
        if isinstance(x, dict) and "prompt" in x and str(x["prompt"]).strip():
            prompts.append(str(x["prompt"]).strip())
    return prompts


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
    payload = enforce_employee_email(payload, DEFAULT_EMPLOYEE_EMAIL)

    try:
        if method == "GET":
            response = SESSION.get(
                url,
                headers=headers,
                params=payload if isinstance(payload, dict) else {"Query": user_prompt},
                timeout=REQUEST_TIMEOUT,
                verify=VERIFY_SSL,
            )
        else:
            response = SESSION.request(
                method,
                url,
                headers=headers,
                json=payload if isinstance(payload, (dict, list)) else {"Query": user_prompt},
                timeout=REQUEST_TIMEOUT,
                verify=VERIFY_SSL,
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
            "response_text": api_text,
        }

    except requests.exceptions.RequestException as e:
        return {
            "status_code": -1,
            "request_body": payload,
            "response_text": f"API_ERROR: {e}",
        }


# =========================================================
# 10) LLM-B: SOP-BASED VALIDATION (NO EXPECTED COMPARISON)
# =========================================================
def validate_response_against_sop(
    user_prompt: str,
    api_response_text: str,
    api_doc_text: str,
    sop_text: str,
) -> Dict[str, str]:
    """
    Evaluator directly judges if API response is correct per SOP + API docs.
    Returns strict verdict + reason + source details.
    """
    actual_msg = extract_message_text(api_response_text)

    p = f"""
You are a policy compliance evaluator.

Task:
Judge whether the ACTUAL API RESPONSE is correct for the USER PROMPT,
strictly based on SOP corpus and API documentation.

Decision rules:
1) Verdict must be only "Yes" or "No".
2) "Yes" only if response is aligned with SOP policy/process and API intent.
3) "No" if response violates SOP, misses required behavior, is out-of-scope, or is incorrect.
4) Reason must be short, precise, and evidence-based.
5) Source must cite specific SOP document/rule and/or API doc section.

USER PROMPT:
{user_prompt}

ACTUAL API RESPONSE:
{actual_msg}

API DOCUMENTATION:
{truncate_text(api_doc_text, 18000)}

SOP CORPUS:
{truncate_text(sop_text, 18000)}

Return ONLY valid JSON:
{{
  "verdict": "Yes|No",
  "reason": "exact reason for verdict",
  "source": "SOP:<file/rule> + API_DOC:<section>"
}}
"""
    out = llm_chat(MODEL_EVAL, p).strip()

    def normalize_verdict(raw: str) -> Optional[str]:
        v = (raw or "").strip().lower()
        if not v:
            return None
        yes_words = {"yes", "pass", "passed", "match", "matched", "true", "acceptable", "aligned"}
        no_words = {"no", "fail", "failed", "mismatch", "false", "not acceptable", "not aligned"}
        if v in yes_words:
            return "Yes"
        if v in no_words:
            return "No"
        if re.search(r"\b(yes|pass|passed|match|matched|acceptable|aligned|true)\b", v):
            return "Yes"
        if re.search(r"\b(no|fail|failed|mismatch|false|not\s+acceptable|not\s+aligned)\b", v):
            return "No"
        return None

    try:
        parsed = parse_json_safely(out)
        if isinstance(parsed, dict):
            verdict = normalize_verdict(str(parsed.get("verdict", "")))
            if verdict is None:
                verdict = normalize_verdict(str(parsed.get("result", "")))
            if verdict is None:
                verdict = normalize_verdict(str(parsed.get("label", "")))
            if verdict is None:
                verdict = "No"

            reason = str(parsed.get("reason", "")).strip() or "No reason provided by evaluator."
            source = str(parsed.get("source", "")).strip() or "SOP + API_DOC"
            return {"verdict": verdict, "reason": reason, "source": source}
    except Exception:
        pass

    plain_verdict = normalize_verdict(out)
    if plain_verdict is not None:
        return {
            "verdict": plain_verdict,
            "reason": "Evaluator returned plain-text verdict without structured reason.",
            "source": "SOP + API_DOC",
        }

    return {
        "verdict": "Evaluation Failed",
        "reason": "Evaluator returned malformed output.",
        "source": "EVALUATOR_OUTPUT_PARSE_FAILED",
    }


# =========================================================
# 11) ORCHESTRATION
# =========================================================
def pick_topics(api_meta: Dict[str, Any]) -> List[str]:
    intents = api_meta.get("test_intents", [])
    topics = [x.strip() for x in intents if isinstance(x, str) and x.strip()]
    if topics:
        return topics
    return ["General Functional Validation", "Error Handling", "Policy Validation"]


def run_test_cycle(
    api_meta: Dict[str, Any],
    runtime_cfg: Dict[str, Any],
    doc_text: str,
    sop_text: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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

            seed = generate_seed_question(topic, api_meta, doc_text, sop_text)
            seed_call = call_target_api(seed, runtime_cfg)

            if seed_call["status_code"] == 401:
                raise PermissionError("401 Unauthorized during seed call. Check auth header/signature/body contract.")

            try:
                fups = (
                    generate_followups(
                        seed,
                        seed_call["response_text"],
                        api_meta,
                        doc_text,
                        sop_text,
                    )
                    if FOLLOWUPS_PER_SEED > 0
                    else []
                )
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

    for idx, prompt in enumerate(all_prompts, start=1):
        print(f"\nRunning Test Case {idx}/{len(all_prompts)}")

        try:
            api_call = call_target_api(prompt, runtime_cfg)
            status_code = api_call["status_code"]
            request_body = api_call["request_body"]
            api_response = api_call["response_text"]
        except Exception as e:
            status_code = -1
            request_body = {}
            api_response = f"API_ERROR: {e}"

        if status_code == 200:
            try:
                eval_result = validate_response_against_sop(
                    user_prompt=prompt,
                    api_response_text=api_response,
                    api_doc_text=doc_text,
                    sop_text=sop_text,
                )
                verdict = eval_result.get("verdict", "Evaluation Failed")
                reason = eval_result.get("reason", "No reason provided by evaluator.")
                source = eval_result.get("source", "SOP + API_DOC")
            except Exception as e:
                verdict = "Evaluation Failed"
                reason = f"Evaluator exception: {e}"
                source = "EVALUATOR_EXCEPTION"
        else:
            verdict = "No"
            reason = f"API returned non-200 status code: {status_code}."
            source = "API_RESPONSE_STATUS"

        worked = verdict if verdict in {"Yes", "No", "Evaluation Failed"} else "Evaluation Failed"

        results.append(
            {
                "Test ID": idx,
                "Prompt": prompt,
                "API Response": extract_message_text(api_response),
                "Request Body": json.dumps(request_body, ensure_ascii=False),
                "Comparison Result": verdict,   # keep for compatibility
                "Reason": reason,               # exact reason for Yes/No
                "Worked As Intended": worked,   # final correctness based on SOP
                "Source": source,               # document/rule citation
            }
        )

    diagnostics = {
        "seed_success_count": seed_success,
        "total_generated_prompts": len(all_prompts),
        "resolved_url": runtime_cfg["url"],
        "resolved_method": runtime_cfg["method"],
        "evaluation_mode": "SOP_DIRECT_VALIDATION_NO_EXPECTED_OUTPUT",
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
    summary = pd.DataFrame(
        [
            {
                "Total Tests": total,
                "Passed": passed,
                "Failed": failed,
                "Evaluation Failed": eval_failed,
                "Success Rate (%)": success,
            }
        ]
    )
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
        raise ValueError("Set API_DOC_PATH in .env.local (mandatory).")
    if not SOP_FOLDER_PATH:
        raise ValueError("Set SOP_FOLDER_PATH in .env.local for this experiment (mandatory).")
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY or not AZURE_OPENAI_VERSION:
        raise ValueError("Set AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_KEY / AZURE_OPENAI_VERSION.")
    if not MODEL_GEN or not MODEL_EVAL:
        raise ValueError("Set MODEL_GEN and MODEL_EVAL.")


def main() -> None:
    validate_config()

    print("\nLoading API documentation...")
    doc_text = load_document_text(API_DOC_PATH)

    print("Loading SOP corpus...")
    sop_text, sop_diag = load_sop_corpus(SOP_FOLDER_PATH, MAX_SOP_FILES, MAX_SOP_CHARS)
    if not sop_diag.get("sop_provided", False):
        raise ValueError(
            f"SOP corpus is required for this experiment. Details: {sop_diag.get('sop_note', 'unknown')}"
        )

    print(
        json.dumps(
            {
                "sop_folder": SOP_FOLDER_PATH,
                "sop_enabled": True,
                "sop_loaded_files": sop_diag.get("sop_loaded_files", 0),
                "sop_skipped_files": sop_diag.get("sop_skipped_files", 0),
                "sop_total_chars": sop_diag.get("sop_total_chars", 0),
                "sop_note": sop_diag.get("sop_note", ""),
            },
            indent=2,
        )
    )

    print("Extracting API metadata from documentation...")
    api_meta = extract_api_config_from_doc(doc_text)

    print("Building runtime API configuration...")
    runtime_cfg = build_runtime_api_config(api_meta)

    print("\nResolved API config:")
    print(
        json.dumps(
            {
                "method": runtime_cfg["method"],
                "url": runtime_cfg["url"],
                "headers": mask_secret_headers(runtime_cfg["headers"]),
                "query_fields": runtime_cfg["query_fields"],
                "required_fields": runtime_cfg["required_fields"],
            },
            indent=2,
        )
    )

    results, diagnostics = run_test_cycle(
        api_meta=api_meta,
        runtime_cfg=runtime_cfg,
        doc_text=doc_text,
        sop_text=sop_text,
    )
    diagnostics.update(sop_diag)
    save_report(results, diagnostics, OUTPUT_EXCEL)


if __name__ == "__main__":
    main()