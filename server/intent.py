"""Turn a typed request ("energetic house, 45 seconds from the chorus at 1:10")
into dance settings, using a small LLM that runs on this machine.

The model is Qwen2.5-0.5B-Instruct (GGUF, about 400 MB) via llama.cpp, on CPU.
Its only job is one function call, set_dance(...), and its output is
constrained to that function's JSON schema with a grammar, so even a tiny
model can't produce malformed arguments. If the model is missing, slow or
wrong, a rule-based parser fills in, so the app always answers.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from choreographer import GENRES

MODEL_PATH = Path(os.environ.get("LLM_PATH", Path(__file__).resolve().parent.parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"))

STYLE_CODES = list(GENRES)
STYLE_WORDS = {
    "BR": ["break", "breaking", "bboy", "b-boy", "breakdance"], "PO": ["pop", "popping"],
    "LO": ["lock", "locking"], "MH": ["middle hip", "middle hip-hop", "middle hiphop", "hip hop", "hip-hop", "hiphop"],
    "LH": ["la hip", "la-style", "la style"], "HO": ["house"], "WA": ["waack", "whack", "wack", "voguing", "vogue"],
    "KR": ["krump"], "JS": ["street jazz", "jazz"], "JB": ["ballet", "ballet jazz", "graceful", "elegant"],
}

SET_DANCE = {
    "name": "set_dance",
    "description": "Configure the dance to generate for the user's uploaded song.",
    "parameters": {
        "type": "object",
        "properties": {
            "style": {"type": "string", "enum": STYLE_CODES + ["auto"],
                      "description": "Dance style code: " + ", ".join(f"{k}={v}" for k, v in GENRES.items()) + ", auto=let the system choose"},
            "seconds": {"type": "integer", "minimum": 5, "maximum": 180, "description": "Length of the dance in seconds"},
            "start": {"type": "number", "minimum": 0, "description": "Where in the song to start, in seconds"},
            "energy": {"type": "string", "enum": ["calm", "match", "intense"],
                       "description": "calm = softer moves, intense = bigger moves, match = follow the music"},
            "reply": {"type": "string", "description": "One short friendly sentence telling the user what you chose"},
        },
        "required": ["style", "seconds", "start", "energy", "reply"],
    },
}

SYSTEM = (
    "You are the choreographer assistant in a song-to-dance app. The user has uploaded a song and describes the dance "
    "they want. Always respond by calling set_dance with JSON arguments. Map the request to the closest style; use "
    "'auto' only if no style is implied. Convert times like 1:10 to seconds (70). Default seconds=30, start=0, "
    "energy=match. Keep reply under 20 words.\n\nFunction:\n" + json.dumps(SET_DANCE)
)

_llm = None
_llm_lock = threading.Lock()
_llm_error: str | None = None


def _load_llm():
    global _llm, _llm_error
    if _llm is not None:
        return _llm
    if not MODEL_PATH.exists():
        # Not downloaded (yet): check again next time, the Space fetches it in the background.
        _llm_error = f"no model at {MODEL_PATH}"
        return None
    if _llm_error is not None and not _llm_error.startswith("no model"):
        return None
    try:
        from llama_cpp import Llama
        _llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, n_threads=max(1, os.cpu_count() or 1), verbose=False)
    except Exception as e:  # missing package or weights: rules only
        _llm_error = str(e)
    return _llm


def llm_status() -> str:
    return "ready" if _load_llm() is not None else f"unavailable ({_llm_error})"


def _clock_to_seconds(m: re.Match) -> float:
    return int(m.group(1)) * 60 + int(m.group(2))


def parse_rules(text: str) -> dict:
    """Keyword parser: the fallback, and a sanity check on the model."""
    t = text.lower()
    out: dict = {"style": "auto", "seconds": 30, "start": 0.0, "energy": "match"}
    best = None
    for code, words in STYLE_WORDS.items():
        for w in words:
            i = t.find(w)
            if i >= 0 and (best is None or len(w) > best[1]):
                best = (code, len(w))
    if best:
        out["style"] = best[0]
    if re.search(r"\b(calm|chill|soft|slow|gentle|relaxed|smooth|graceful)\b", t):
        out["energy"] = "calm"
    if re.search(r"\b(energetic|intense|hype|hard|wild|crazy|powerful|aggressive|big|explosive|fast)\b", t):
        out["energy"] = "intense"
    # start: "from 1:10", "at 70s", "starting at 45 seconds"
    m = re.search(r"(?:from|at|starting(?: at)?)\s+(\d+):(\d{2})", t)
    if m:
        out["start"] = float(_clock_to_seconds(m))
    else:
        m = re.search(r"(?:from|at|starting(?: at)?)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)\b", t)
        if m:
            out["start"] = float(m.group(1))
    # length: "45 seconds", "1 minute", "a minute and a half", "2 min"
    t_len = re.sub(r"(?:from|at|starting(?: at)?)\s+\d+(?::\d{2}|(?:\.\d+)?\s*(?:s|sec|secs|seconds)\b)", " ", t)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b", t_len)
    if m:
        out["seconds"] = int(float(m.group(1)) * 60)
    elif re.search(r"\b(a|one) minute and a half\b", t_len):
        out["seconds"] = 90
    elif re.search(r"\b(a|one) minute\b", t_len):
        out["seconds"] = 60
    else:
        m = re.search(r"(\d+)\s*(?:s|sec|secs|seconds)\b", t_len)
        if m:
            out["seconds"] = int(m.group(1))
    if re.search(r"\b(whole|full|entire)\s+song\b", t):
        out["seconds"] = 180
    out["seconds"] = max(5, min(180, out["seconds"]))
    return out


def _reply_for(s: dict) -> str:
    mood = {"calm": "gentle ", "intense": "high-energy ", "match": ""}[s["energy"]]
    style = "" if s["style"] == "auto" else GENRES[s["style"]] + " "
    start = f", starting at {int(s['start']) // 60}:{int(s['start']) % 60:02d}" if s["start"] else ""
    pick = " I'll pick the style." if s["style"] == "auto" else ""
    return f"Making a {s['seconds']}-second {mood}{style}dance{start}.{pick}"


def interpret(text: str) -> dict:
    """Return settings plus a reply, and which engine produced them."""
    rules = parse_rules(text)
    llm = None  # settings only now; the AI choreographer (choreo_llm.py) writes the plan
    if llm is not None:
        try:
            with _llm_lock:
                r = llm.create_chat_completion(
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text[:500]}],
                    response_format={"type": "json_object", "schema": SET_DANCE["parameters"]},
                    temperature=0.1, max_tokens=120,
                )
            args = json.loads(r["choices"][0]["message"]["content"])
            s = {
                "style": args.get("style") if args.get("style") in STYLE_CODES + ["auto"] else rules["style"],
                "seconds": int(max(5, min(180, int(args.get("seconds", rules["seconds"]))))),
                "start": float(max(0.0, float(args.get("start", rules["start"])))),
                "energy": args.get("energy") if args.get("energy") in ("calm", "match", "intense") else rules["energy"],
            }
            # Trust anything the keyword parser found explicitly over a tiny model's guess.
            defaults = {"style": "auto", "seconds": 30, "start": 0.0, "energy": "match"}
            overridden = False
            for k, d in defaults.items():
                if rules[k] != d and s[k] != rules[k]:
                    s[k] = rules[k]
                    overridden = True
            s["reply"] = _reply_for(s) if overridden or not args.get("reply") else str(args["reply"])[:200]
            s["parser"] = "llm"
            return s
        except Exception:
            pass
    rules["reply"] = _reply_for(rules)
    rules["parser"] = "rules"
    return rules
