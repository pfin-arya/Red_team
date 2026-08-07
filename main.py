import json
import logging
import os
import re
import time
import tkinter as tk
from tkinter import messagebox
import warnings

import docx
import pandas as pd
import requests
from docx import Document
from requests.packages.urllib3.exceptions import InsecureRequestWarning

from generator import generate_and_save_question_answers

logging.basicConfig(level=logging.INFO)

# Silence verify=False warning if you intentionally allow insecure SSL.
warnings.simplefilter("ignore", InsecureRequestWarning)

DEFAULT_API_URL = os.getenv("PAI_CHAT_URL", "")
API_TIMEOUT_SEC = int(os.getenv("API_TIMEOUT_SEC", "20"))      # reduced from 120
MAX_ROWS = int(os.getenv("MAX_ROWS", "5"))                    # run fewer rows by default for speed
REPORT_PATH = os.getenv("REPORT_PATH", "Chatbot_3_Column_Report.xlsx")

def markdownize(doc: Document) -> str:
    text = ""
    for element in doc.element.body:
        if element.tag.endswith("p"):
            para = docx.text.paragraph.Paragraph(element, doc)
            if para.text.strip():
                text += para.text + "\n\n"
        elif element.tag.endswith("tbl"):
            table = docx.table.Table(element, doc)
            if table.rows:
                headers = [cell.text.strip() for cell in table.rows[0].cells]
                text += "| " + " | ".join(headers) + " |\n"
                text += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for row in table.rows[1:]:
                    cells = [cell.text.strip() for cell in row.cells]
                    text += "| " + " | ".join(cells) + " |\n"
                text += "\n"
    return text


def read_doc_text(path: str) -> str:
    if path.lower().endswith(".docx"):
        return markdownize(Document(path))
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_api_url(text: str) -> str:
    urls = re.findall(r"https?://[^\s'\"<>]+", text or "")
    if not urls:
        return ""
    preferred = [u for u in urls if "/api/" in u.lower()]
    return preferred[0] if preferred else urls[0]


def build_request_payload(question_row: dict) -> dict:
    message = (
        question_row.get("message")
        or question_row.get("query")
        or question_row.get("question")
        or question_row.get("prompt")
        or question_row.get("input")
        or "Hello"
    )
    employee_email = (
        question_row.get("employee_email")
        or question_row.get("email")
        or question_row.get("userEmail")
        or "test.user@company.com"
    )
    channel = question_row.get("channel_name") or "teams"

    return {
        "employee_email": str(employee_email),
        "message": str(message),
        "channel": {"name": str(channel)},
    }


def call_api(url: str, payload: dict) -> dict:
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=API_TIMEOUT_SEC,
        verify=False,
    )
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return {"raw_response": response.text}


def pick_question_value(row: dict) -> str:
    for key in ["question", "query", "message", "prompt", "input", "user_query", "user_input"]:
        if key in row and pd.notna(row[key]):
            s = str(row[key]).strip()
            if not s:
                continue
            # Avoid selecting answer-like long paragraphs as question
            if len(s) > 220 and ("." in s or "\n" in s):
                continue
            return s
    return "N/A"


def pick_expected_answer_value(output_row: dict, question_text: str = "") -> str:
    # Expected answer must come only from output-like fields.
    preferred_output_keys = [
        "answer",
        "expected_answer",
        "response_text",
        "output_text",
        "assistant_response",
        "chatbot_response",
        "final_answer",
        "resolution",
        "response",
        "output",
        "message",  # keep last, only if truly output message
    ]

    q_norm = (question_text or "").strip().lower()

    # 1) Strong priority keys
    for key in preferred_output_keys:
        if key not in output_row:
            continue

        val = output_row.get(key)
        if pd.isna(val):
            continue
        if isinstance(val, (dict, list, tuple, set)):
            continue

        s = str(val).strip()
        if not s:
            continue

        s_norm = s.lower()
        if q_norm and s_norm == q_norm:
            continue
        if "_" in s and " " not in s:  # label-like token
            continue
        if s in {"[]", "{}"}:
            continue
        if s_norm in {
            "n/a",
            "na",
            "null",
            "none",
            "letter_generation_agent",
            "general_conversation_agent",
            "reimbursement_status_query",
        }:
            continue
        return s

    # 2) Generic fallback from output row only (never question-like keys)
    banned_key_tokens = {
        "question", "query", "prompt", "input", "intent", "agent", "route",
        "channel", "tenant", "email", "id", "code", "status", "type"
    }

    best = ""
    for k, v in output_row.items():
        if pd.isna(v):
            continue
        if isinstance(v, (dict, list, tuple, set)):
            continue

        kl = str(k).strip().lower()
        if any(tok in kl for tok in banned_key_tokens):
            continue

        s = str(v).strip()
        if not s:
            continue
        if s in {"[]", "{}"}:
            continue

        s_norm = s.lower()
        if q_norm and s_norm == q_norm:
            continue
        if "_" in s and " " not in s:
            continue
        if s_norm in {
            "n/a",
            "na",
            "null",
            "none",
            "letter_generation_agent",
            "general_conversation_agent",
            "reimbursement_status_query",
        }:
            continue

        # Prefer answer-like sentence text.
        if len(s) >= 20 and len(s) > len(best):
            best = s

    return best if best else "N/A"


def submit():
    api_doc_path = entry_api_doc.get().strip()
    synth_context_file = entry_synth_context_file.get().strip()
    synth_domain = entry_synth_domain.get().strip()
    synth_count_raw = entry_synth_count.get().strip() or "25"
    sop_folder_path = entry_sop_folder.get().strip()

    if not api_doc_path:
        messagebox.showerror("Missing Input", "Please provide API documentation path.")
        return
    if not os.path.exists(api_doc_path):
        messagebox.showerror("Invalid Path", "API documentation file does not exist.")
        return

    try:
        synth_count = int(synth_count_raw)
        if synth_count <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Invalid Input", "Synthetic rows count must be a positive integer.")
        return

    try:
        api_doc_text = read_doc_text(api_doc_path)
    except Exception as exc:
        messagebox.showerror("Read Error", f"Unable to read API documentation: {exc}")
        return

    api_url = extract_api_url(api_doc_text) or DEFAULT_API_URL
    if not api_url:
        messagebox.showerror(
            "API URL Not Found",
            "Could not find API endpoint in document and PAI_CHAT_URL env var is empty.",
        )
        return

    t0 = time.time()
    try:
        input_xlsx, output_xlsx = generate_and_save_question_answers(
            count=synth_count,
            domain=synth_domain or None,
            context_file=synth_context_file or None,
            api_doc_file=api_doc_path,
            sop_folder=sop_folder_path or None,
        )
    except Exception as exc:
        messagebox.showerror("Generation Failed", f"Failed to generate synthetic Q&A: {exc}")
        return

    try:
        input_df = pd.read_excel(input_xlsx)
        output_df = pd.read_excel(output_xlsx)
    except Exception as exc:
        messagebox.showerror("Read Error", f"Failed to read generated datasets: {exc}")
        return

    result_rows = []

    if MAX_ROWS > 0:
        input_df = input_df.head(MAX_ROWS)
        output_df = output_df.head(MAX_ROWS)

    total_rows = len(input_df)
    if total_rows == 0:
        messagebox.showerror("No Data", "Generated dataset is empty.")
        return

    for idx, (_, in_row) in enumerate(input_df.iterrows(), start=1):
        print(f"[RUNNING] Sr No: {idx}/{total_rows}")
        logging.info("Running row Sr No: %s/%s", idx, total_rows)

        in_dict = in_row.to_dict()
        out_dict = output_df.iloc[idx - 1].to_dict() if idx - 1 < len(output_df) else {}

        payload = build_request_payload(in_dict)

        try:
            api_response = call_api(api_url, payload)
            actual_answer = (
                api_response.get("message")
                or api_response.get("answer")
                or api_response.get("response")
                or json.dumps(api_response, ensure_ascii=False)
            )
        except Exception as exc:
            actual_answer = f"API call failed: {exc}"

        question_text = pick_question_value(in_dict)
        expected_answer = pick_expected_answer_value(out_dict, question_text=question_text)

        result_rows.append(
            {
                "Sr_No": idx,
                "Question": question_text,
                "Expected_Answer": expected_answer,
                "Actual_API_Response": actual_answer,
            }
        )

    report_df = pd.DataFrame(
        result_rows,
        columns=["Sr_No", "Question", "Expected_Answer", "Actual_API_Response"],
    )
    report_df.to_excel(REPORT_PATH, index=False)

    elapsed = round(time.time() - t0, 2)
    messagebox.showinfo(
        "Success",
        f"Report generated successfully.\n"
        f"API URL: {api_url}\n"
        f"Rows processed: {total_rows}\n"
        f"Total time: {elapsed}s\n"
        f"Report: {REPORT_PATH}",
    )
    root.destroy()


root = tk.Tk()
root.title("PFL-ATE")

tk.Label(root, text="API documentation path:").grid(row=0, column=0)
entry_api_doc = tk.Entry(root, width=70)
entry_api_doc.grid(row=0, column=1)

tk.Label(root, text="SOP folder path (optional):").grid(row=1, column=0)
entry_sop_folder = tk.Entry(root, width=70)
entry_sop_folder.grid(row=1, column=1)

tk.Label(root, text="Synthetic context file path (optional):").grid(row=2, column=0)
entry_synth_context_file = tk.Entry(root, width=70)
entry_synth_context_file.grid(row=2, column=1)

tk.Label(root, text="Synthetic domain (optional):").grid(row=3, column=0)
entry_synth_domain = tk.Entry(root, width=70)
entry_synth_domain.grid(row=3, column=1)

tk.Label(root, text="Synthetic rows count:").grid(row=4, column=0)
entry_synth_count = tk.Entry(root, width=70)
entry_synth_count.insert(0, "25")
entry_synth_count.grid(row=4, column=1)

tk.Button(root, text="Run End-to-End", command=submit).grid(row=5, column=1)
root.mainloop()