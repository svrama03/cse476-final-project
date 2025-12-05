import os, re
import requests

API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")  
MODEL    = os.getenv("MODEL_NAME", "bens_model")              

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                max_tokens: int = 128,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

# Helper functions
def _error_response(e: Exception) -> dict:
    return {
        "ok": False,
        "text": "",
        "raw": None,
        "status": -1,
        "error": f"Error calling model: {e}",
        "headers": {},
    }

def direct(question: str) -> dict:
    system_prompt = "You are a helpful assistant, reply with only the final answer and give no explanations."
    try:
        resp = call_model_chat_completions(question, system=system_prompt, temperature=0)
        resp["text"] = extract_final_answer(resp.get("text" or "").strip())
        return resp
    except Exception as e:
        return _error_response(e)
    
def chain_of_thought(question: str) -> dict:
    cot_prompt = f"""
    You are a careful reasoning assistant.
    Question: {question}
    First, think through the problem step-by-step. Then, on a new line, write:
    Final answer: <your short final answer here>""".strip()

    try:
        resp = call_model_chat_completions(cot_prompt, max_tokens=256, temperature=0.0)
        resp["text"] = extract_final_answer(resp.get("text" or "").strip())
        return resp
    except Exception as e:
        return _error_response(e)
    
def self_refine(question: str, initial_answer: str) -> dict:
    self_refine_prompt = f"""You are a careful, self-checking assistant.
    Question: {question}
    Initial Answer: {initial_answer}
    1. Check whether the initial answer is logically and mathematically correct.
    2. If it is correct, keep it.
    3. If it is not correct, fix it.
    Write any reasoning you need, then on a new line write: Final answer: <your corrected short final answer>""".strip()
    try:
        resp = call_model_chat_completions(self_refine_prompt, max_tokens=512, temperature=0)
        resp["text"] = extract_final_answer(resp.get("text" or "").strip())
        return resp
    except Exception as e:
        return _error_response(e)
    
def reasoning_strategy(question: str) -> dict:
    cot_resp = chain_of_thought(question)
    if not cot_resp.get("ok", False):
        return direct(question)

    try:
        refine_resp = self_refine(question, cot_resp.get("text", ""))
    except Exception:
        return cot_resp
            
    if refine_resp.get("ok", False):
        return refine_resp
    else: 
        return cot_resp
    
def coding(question: str) -> dict:
    system_prompt = "You are a Python coding assistant. Return ONLY valid Python code that solves the task, no " \
    "explanations, no comments outside the code blocks."
    try:
        resp = call_model_chat_completions(question, system=system_prompt, max_tokens=512, temperature=0.0)
        resp["text"] = resp.get("text" or "").strip()
        return resp
    except Exception as e:
        return _error_response(e)
    
def prediction(question: str) -> dict:
    system_prompt = "You are an assistant that predicts future events. You MUST make a single clear prediction and " \
    "ensure the final line ends with exactly one LaTeX-style box: [{YOUR_PREDICTION}]."

    try:
        resp = call_model_chat_completions(question, system=system_prompt, max_tokens=512, temperature=0.0)
        resp["text"] = resp.get("text" or "").strip()
        return resp
    except Exception as e:
        return _error_response(e)

def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"Final answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# Keyword lists for question type guessing
PREDICTION_KEYWORDS = (
    "predict", "forecast", "will happen", "future", "in the year", "by 20", "by 202",
    "next decade", "next year", "next month", "next week", "next century",
    "in 5 years", "in 10 years", "in 50 years", "in 100 years",
    "trends", "emerging", "developments", "advancements", "evolution",
    "projections", "anticipated", "expected", "likely to", "could be", "might be",
)

CODING_KEYWORDS = (
    "code", "function", "class", "script", "program", "bug", "traceback",
    "syntaxerror", "runtimeerror", "compile error", "compilation error",
    "stack trace",
)

CODING_LANGUAGES = (
    "python", "java", "javascript", "c++", "c#", "rust", "go",
    "typescript", "ruby", "php", "swift", "kotlin", "html", "css", "sql",
)

MATH_KEYWORDS = (
    "calculate", "compute", "evaluate", "what is", "find the value of",
    "determine", "solve", "integrate", "differentiate", "sum of",
    "product of", "plus", "minus", "times", "divided by",
    "equation", "formula", "algebra", "geometry", "trigonometry", "calculus",
)

def guess_question_type(question: str) -> str:
    q = question.lower()

    if any(kw in q for kw in PREDICTION_KEYWORDS):
        return "future_prediction"

    if "```" in q or any(kw in q for kw in CODING_KEYWORDS) or any(lang in q for lang in CODING_LANGUAGES):
        return "coding"

    if any(kw in q for kw in MATH_KEYWORDS):
        return "math"
    
    return "default"

def run_agent(question: str) -> str:
    question_type = guess_question_type(question)

    strategy_map = {
        "coding": coding,
        "future_prediction": prediction,
        "math": reasoning_strategy,
        "default": direct
    }

    strategy = strategy_map.get(question_type, strategy_map["default"])
    resp = strategy(question)

    if not resp.get("ok", False):
        resp = direct(question)

    return resp.get("text", "")