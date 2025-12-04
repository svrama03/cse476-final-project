# %% Minimal setup
# If needed (uncomment in a notebook):
# !pip install requests python-dotenv

import os, json, textwrap, re, time
import requests
from typing import Tuple, Optional

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

def direct(question: str) -> Optional[str]:
    system_prompt = "You are a helpful assistant, reply with only the final answer and give no explanations."
    try:
        resp = call_model_chat_completions(question, system=system_prompt, temperature=0)
        return resp.strip()
    except Exception as e:
        print(f"Error calling model: {e}")
        return None
    
def chain_of_thought(question: str) -> Optional[str]:
    cot_prompt = f"""You are a helpful assistant. Think through the problem step-by-step before providing the final answer.
    Question: {question}
    Let's approach this systematically."""

    try:
        resp = call_model_chat_completions(cot_prompt, max_tokens=512, temperature=0.0)
        return resp.strip()
    except Exception as e:
        print(f"Error calling model: {e}")
        return None
    
def self_refine(question: str, initial_answer: str) -> Optional[str]:
    self_refine_prompt = f"""Question: {question}
    Initial Answer: {initial_answer}
    Review the answer above. If it's correct, return it as-is. If there are errors, provide a corrected version."""
    
    try:
        resp = call_model_chat_completions(self_refine_prompt, max_tokens=2048, temperature=0)
        return resp.strip()
    except Exception as e:
        print(f"Error calling model: {e}")
        return None
    
def reasoning_strategy(question: str) -> Tuple[bool, str]:
    """
    Chain of thought followed by self-refinement, utilized in math/planning
    """
    ok_cot, cot_answer = chain_of_thought(question)
    if not ok_cot:
        return direct(question)

    ok_refine, refined = self_refine(question, cot_answer)
    if ok_refine:
        return True, refined
    else:
        return True, cot_answer
    
def coding(question: str) -> Tuple[bool, str]:
    """
    Coding tasks strategy
    """
    systemPrompt = "You are a Python coding assistant. Return ONLY valid Python code that solves the task, no explanations, no comments outside the code block."
    systemResponse = call_model_chat_completions(question, system=systemPrompt, max_tokens=2048, temperature=0.0)
    if systemResponse.get('ok'):
        return True, systemResponse.get('text', '').strip()
    else:
        return False, systemResponse.get('error')
    
def prediction(question: str) -> Tuple[bool, str]:
    """
    Strategy for future prediction tasks that must end with \boxed{...}.
    """
    systemPrompt = (
        "You are an assistant that predicts future events. "
        "You MUST make a single clear prediction and ensure the final line ends with "
        "exactly one LaTeX-style box: \\boxed{YOUR_PREDICTION}."
    )

    systemResponse = reasoning_strategy(question)

    if systemResponse.get('ok'):
        return True, systemResponse.get('text', '').strip()
    else:
        return False, systemResponse.get('error')

def guess_question_type(question: str) -> str:
    """
    Guesses/Classifies the question type
    """
    q = question.lower()

    if "predict" in q and "\\boxed" in q:
        return "future_prediction"
    if "[plan]" in q and "my plan is as follows" in q:
        return "planning"
    if "```" in q or "def task_func" in q or "from datetime import" in q:
        return "coding"
    return "math"

def run_agent(question: str) -> Tuple[bool, str]:
    """
    Agent Loop
    """
    question_type = guess_question_type(question)

    strategy_map = {
        "coding": coding,
        "future_prediction": prediction,
        "math": reasoning_strategy,
        "planning": reasoning_strategy,
        "default": direct
    }

    strategy = strategy_map.get(question_type, strategy_map["default"])
    ok, answer = strategy(question)

    if not ok:
        ok, answer = direct(question)

    return ok, answer