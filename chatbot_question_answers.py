"""
Synthetic Chatbot Question-Answer Generator.

Generates synthetic chatbot data based on user-provided API input/output format.
Produces two XLSX files:
  - ate_chatbot_ip.xlsx  (columns matching the API input schema)
  - ate_chatbot_op.xlsx  (columns matching the API output schema)

The module loads API schema JSON from:
    - chatbot_context/config/api_input_schema.txt
    - chatbot_context/config/api_output_schema.txt
and generates synthetic rows matching those exact schemas.

Optionally accepts context files (PDF, DOCX, Excel, CSV) to inform the generation.

Usage:
    python chatbot_question_answers.py --count 25
    python chatbot_question_answers.py --count 25 --domain banking --context-file policy.pdf
"""

import argparse
import json
import logging
import os

import openai
import pandas as pd
from dotenv import load_dotenv
from synth_logger import synth_logger, wire_python_logging

load_dotenv()
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("OPENAI")
MODEL_NAME = "gpt-4o"

OUTPUT_DIR = "chatbot_question_answers"
BASE_DIR = os.path.dirname(__file__)
CHATBOT_CONFIG_DIR = os.path.join(BASE_DIR, "chatbot_context", "config")
API_INPUT_SCHEMA_PATH = os.path.join(CHATBOT_CONFIG_DIR, "api_input_schema.txt")
API_OUTPUT_SCHEMA_PATH = os.path.join(CHATBOT_CONFIG_DIR, "api_output_schema.txt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
wire_python_logging()


def _get_client() -> openai.AzureOpenAI:
    return openai.AzureOpenAI(
        api_version=OPENAI_API_VERSION,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


def _extract_context_from_file(file_path: str) -> str:
    """Extract text content from a context file (PDF, DOCX, XLSX, CSV)."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
        return f"CSV data summary:\nColumns: {list(df.columns)}\nSample rows:\n{df.head(10).to_string()}"

    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
        return f"Excel data summary:\nColumns: {list(df.columns)}\nSample rows:\n{df.head(10).to_string()}"

    elif ext == ".pdf":
        try:
            import PyPDF2
            text_parts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages[:20]:  # Limit to first 20 pages
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)[:15000]  # Limit context size
        except ImportError:
            logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
            return ""

    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(file_path)
            text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n".join(text_parts)[:15000]
        except ImportError:
            logger.warning("python-docx not installed. Install with: pip install python-docx")
            return ""

    else:
        logger.warning("Unsupported context file format: %s", ext)
        return ""


def _load_json_schema_from_file(path: str, schema_name: str) -> dict:
    """Load a JSON schema object from a required file path."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find {schema_name} schema file. Expected at: {path}"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to load {schema_name} schema from {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid {schema_name} schema at {path}: JSON root must be an object")

    logger.info("Loaded %s schema from: %s", schema_name, path)
    return payload


def generate_question_answers(
    count: int = 25,
    domain: str | None = None,
    context: str | None = None,
    context_file: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Generate synthetic chatbot data using Azure OpenAI based on API input/output format.

    Parameters
    ----------
    count : int
        Number of rows to generate.
    domain : str, optional
        Domain of the chatbot (e.g. "education loans", "banking", "healthcare").
    context : str, optional
        Additional context about the chatbot.
    context_file : str, optional
        Path to a context file (PDF, DOCX, XLSX, CSV) for generating relevant data.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (inputs, outputs) where each list contains dicts matching the respective schema.
    """
    api_input = _load_json_schema_from_file(API_INPUT_SCHEMA_PATH, "api_input")
    api_output = _load_json_schema_from_file(API_OUTPUT_SCHEMA_PATH, "api_output")

    domain_str = domain if domain else "general-purpose"

    # Build context from file if provided
    file_context = ""
    if context_file and os.path.isfile(context_file):
        file_context = _extract_context_from_file(context_file)
        if file_context:
            file_context = f"\n\nReference document content:\n{file_context}"

    additional_context = ""
    if context:
        additional_context = f"\nAdditional context: {context}"

    input_keys = list(api_input.keys())
    output_keys = list(api_output.keys())

    system_prompt = (
        "You are an expert QA dataset creator for chatbot testing. "
        "You generate realistic data that simulates real user interactions "
        "with a chatbot API. Output ONLY valid JSON — no markdown fences, no explanation."
    )

    user_prompt = f"""Generate exactly {count} synthetic data rows for a **{domain_str}** chatbot.{additional_context}{file_context}

The API input format has these fields: {json.dumps(api_input, indent=2)}
The API output format has these fields: {json.dumps(api_output, indent=2)}

Return a JSON array where each element has ALL of these keys:
{json.dumps(input_keys + output_keys)}

For each row, generate realistic values for every field:
- Input fields ({input_keys}): generate realistic user-side data matching the field semantics.
- Output fields ({output_keys}): generate realistic chatbot response data matching the field semantics.

Requirements:
- Mix simple factual questions, how-to questions, troubleshooting questions, and specific scenario queries.
- Output/answer fields should be detailed and informative (1-5 paragraphs depending on complexity).
- Vary sentence length and tone (formal, casual, terse) in input fields.
- Cover a wide range of sub-topics within the domain.
- Do NOT repeat rows.
- Make all values realistic and domain-specific.
- Each row must have EXACTLY these keys: {json.dumps(input_keys + output_keys)}

Return ONLY the JSON array."""

    client = _get_client()
    logger.info("Requesting %d rows for '%s' chatbot from Azure OpenAI...", count, domain_str)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.9,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0].strip()

    rows = json.loads(raw)

    # Split into input and output lists based on schemas
    inputs = []
    outputs = []
    for row in rows:
        inputs.append({k: row.get(k, "") for k in input_keys})
        outputs.append({k: row.get(k, "") for k in output_keys})

    return inputs, outputs


def save_to_xlsx(
    inputs: list[dict],
    outputs: list[dict],
    output_dir: str = OUTPUT_DIR,
    file_index: int = 1,
) -> tuple[str, str]:
    """
    Save Q&A pairs to ate_chatbot_ip.xlsx and ate_chatbot_op.xlsx files.

    Returns
    -------
    tuple[str, str]
        Paths to the input and output xlsx files.
    """
    os.makedirs(output_dir, exist_ok=True)

    ip_filename = f"ate_chatbot_ip_{file_index}.xlsx"
    op_filename = f"ate_chatbot_op_{file_index}.xlsx"
    ip_path = os.path.join(output_dir, ip_filename)
    op_path = os.path.join(output_dir, op_filename)

    # Create input DataFrame
    df_input = pd.DataFrame(inputs)
    df_input.to_excel(ip_path, index=False, engine="openpyxl")
    logger.info("Saved input file: %s", ip_path)

    # Create output DataFrame
    df_output = pd.DataFrame(outputs)
    df_output.to_excel(op_path, index=False, engine="openpyxl")
    logger.info("Saved output file: %s", op_path)

    return ip_path, op_path


def generate_and_save_question_answers(
    count: int = 25,
    domain: str | None = None,
    context: str | None = None,
    context_file: str | None = None,
    file_index: int = 1,
) -> tuple[str, str]:
    """
    Generate data based on API input/output format and save to XLSX files. Returns paths to both files.
    """
    synth_logger.set_context(
        {
            "component": "chatbot_question_answers",
            "count": count,
            "domain": domain,
            "context_file": context_file,
            "file_index": file_index,
        }
    )
    synth_logger.info(
        "Chatbot question-answers generation started",
        extra={"count": count, "domain": domain or "general", "file_index": file_index},
    )

    try:
        inputs, outputs = generate_question_answers(
            count=count,
            domain=domain,
            context=context,
            context_file=context_file,
        )
        ip_path, op_path = save_to_xlsx(inputs, outputs, file_index=file_index)
        synth_logger.info(
            "Chatbot question-answers generation completed",
            extra={"input_path": ip_path, "output_path": op_path},
        )
        synth_logger.flush_sync(status="Success")
        return ip_path, op_path
    except Exception as exc:
        synth_logger.error(
            "Chatbot question-answers generation failed",
            extra={"count": count, "domain": domain or "general", "file_index": file_index},
            exc_info=True,
        )
        synth_logger.flush_sync(status="Failure", error_message=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic chatbot Q&A pairs.")
    parser.add_argument("--count", type=int, default=25, help="Number of rows to generate (default 25).")
    parser.add_argument("--domain", type=str, default=None, help="Domain of chatbot (e.g. education, banking).")
    parser.add_argument("--context", type=str, default=None, help="Additional context about the chatbot.")
    parser.add_argument("--context-file", type=str, default=None, help="Path to context file (PDF, DOCX, XLSX, CSV).")
    args = parser.parse_args()

    ip_path, op_path = generate_and_save_question_answers(
        count=args.count,
        domain=args.domain,
        context=args.context,
        context_file=args.context_file,
    )
    print(f"\nGenerated {args.count} rows:")
    print(f"  Input file:  {ip_path}")
    print(f"  Output file: {op_path}")
