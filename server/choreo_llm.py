"""The AI choreographer: a small LLM writes the dance plan, the retrieval engine performs it.

Given the song's structure (sections found from its energy curve) and what the user
asked for ("a shy robot who goes wild at the drop"), Qwen2.5-Instruct decides, for
each section, the dance style, the energy, a move idea and a one-line reason. The
user can then refine the plan in words ("make the ending calmer").

Small models are unreliable at free-form JSON, so decoding is constrained: the JSON
skeleton is fed to the model, style and energy are chosen only among valid options
(a token trie), and free-text fields stop at the closing quote. Output is always
valid. If the model is missing or fails, a rule-based plan that follows the music
takes over, so the app always answers.

Runs with Hugging Face transformers on CPU (Qwen2.5-0.5B-Instruct, about 1 GB),
or a larger model on GPU via LLM_REPO.
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

from choreographer import GENRES

ROOT = Path(__file__).resolve().parent.parent
LLM_REPO = os.environ.get("LLM_REPO", "Qwen/Qwen2.5-0.5B-Instruct")
LLM_DIR = Path(os.environ.get("LLM_DIR", ROOT / "models" / LLM_REPO.split("/")[-1].lower()))

STYLE_FEEL = {
    "BR": "Break: floor spins, freezes, athletic", "PO": "Pop: robotic hits, isolations, sharp",
    "LO": "Lock: funky, playful points and locks", "MH": "Middle Hip-hop: bouncy groove, confident",
    "LH": "LA Hip-hop: hard-hitting, swagger", "HO": "House: fast footwork, light, joyful",
    "WA": "Waack: arm whips, poses, diva", "KR": "Krump: aggressive, explosive, raw emotion",
    "JS": "Street Jazz: sassy, sharp lines", "JB": "Ballet Jazz: graceful, flowing, elegant",
}
NAME_TO_CODE = {v: k for k, v in GENRES.items()}
ENERGY_WORDS = {"calm": "calm", "steady": "match", "intense": "intense"}
STYLE_WORDS = {
    "BR": ["break", "breaking", "bboy", "b-boy", "breakdance"], "PO": ["pop", "popping", "robot"],
    "LO": ["lock", "locking"], "MH": ["middle hip", "hip hop", "hip-hop", "hiphop"],
    "LH": ["la hip", "la-style", "la style"], "HO": ["house"], "WA": ["waack", "whack", "voguing", "vogue"],
    "KR": ["krump"], "JS": ["street jazz", "jazz"], "JB": ["ballet", "graceful", "elegant"],
}

SYSTEM = (
    "You are an expert AI choreographer. You plan a dance for a song, section by section, "
    "matching the user's request and the music. Dance styles you can use:\n"
    + "\n".join(f"- {v}" for v in STYLE_FEEL.values())
    + "\nEnergy is calm, steady or intense. Quiet music suits calm, loud music suits intense, "
    "unless the user's story says otherwise. Change style between sections only when it helps the story. "
    "Write a short move idea (2-5 words, your own words) and a reason under 12 words.\n\n"
    "Here is an example of the format only; never copy its words.\n"
    "Request: a thunderstorm that rolls in and passes. Plan: "
    '{"title": "Storm Front", "sections": ['
    '{"section": 1, "time": "0:00-0:10", "music": "loud", "style": "Krump", "energy": "intense", '
    '"move": "stomps like thunder", "why": "The storm hits hard, so every step lands heavy."}, '
    '{"section": 2, "time": "0:10-0:20", "music": "quiet", "style": "Ballet Jazz", "energy": "calm", '
    '"move": "arms like drifting rain", "why": "After the storm, the body softens and floats."}], '
    '"reply": "Thunder first, then gentle rain: a storm told in two moods."}'
)

# Words that hint at a style or an energy; they nudge the model's choice (log-prob bonus).
STYLE_HINTS = {
    "KR": r"battle|aggress|angry|anger|rage|fight|warrior|raw|furious|beast",
    "JB": r"elegan|grace|dream|rain|ballerina|swan|flow|soft|tender|float|romantic",
    "PO": r"robot|machine|glitch|android|cyborg|mechanical",
    "LO": r"funk|funny|playful|comic|cartoon|silly|70s|disco",
    "HO": r"happy|joy|party|club|summer|free|light",
    "WA": r"diva|fierce|queen|runway|fashion|pose|drama",
    "BR": r"acrobat|spin|street battle|athlet|power move|gravity",
    "LH": r"swag|gangster|tough|cool|attitude|street",
    "JS": r"sassy|sexy|sharp|confident|attitude",
    "MH": r"groove|bounce|chill|hip|vibe",
}
ENERGY_HINTS = {"calm": r"calm|soft|gentle|slow|quiet|sad|shy|lonely|tender|dream|chill",
                "intense": r"wild|intense|aggress|explod|explos|crazy|hype|power|hard|fierce|battle|rage|drop|go big|insane"}
TARGETS = {"last": r"\b(end|ending|last|outro|finale|final|finish)\b", "first": r"\b(start|beginning|intro|opening|first)\b",
           "loud": r"\b(chorus|drop|peak|climax|loud|hook)\b", "quiet": r"\b(quiet|verse|calm part|soft part|break)\b",
           "middle": r"\b(middle|mid)\b"}

_lock = threading.Lock()
_state: dict = {"model": None, "tok": None, "error": None, "loading": False, "device": "cpu"}


# ---------------------------------------------------------------- model

def last_error() -> str | None:
    return _state["error"]


def model_ready() -> bool:
    return _state["model"] is not None


def status() -> str:
    if _state["model"] is not None:
        return "ready (GPU)" if _remote is not None else "ready"
    if _state["loading"]:
        return "loading"
    return f"unavailable ({_state['error'] or 'not downloaded yet'})"


def download():
    """Fetch the weights from the Hugging Face Hub (the Space does this on boot)."""
    if (LLM_DIR / "config.json").exists():
        return
    from huggingface_hub import snapshot_download
    snapshot_download(LLM_REPO, local_dir=str(LLM_DIR), allow_patterns=["*.json", "*.safetensors", "*.txt"])


_remote = None  # a function that runs _plan_local on a GPU worker (ZeroGPU), set by the Space


def use_remote(fn):
    global _remote
    _remote = fn


def load(device: str | None = None):
    if _state["model"] is not None or _state["loading"]:
        return _state["model"]
    _state["loading"] = True
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not (LLM_DIR / "config.json").exists():
            raise FileNotFoundError(f"no model at {LLM_DIR}")
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        # CPU unless asked: on ZeroGPU Spaces, touching CUDA outside a @spaces.GPU call crashes.
        dev = device or os.environ.get("LLM_DEVICE", "cpu")
        tok = AutoTokenizer.from_pretrained(str(LLM_DIR))
        model = AutoModelForCausalLM.from_pretrained(str(LLM_DIR), dtype=torch.float32 if dev == "cpu" else torch.float16)
        model.to(dev).eval()
        vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
        texts = [tok.convert_tokens_to_string([t]) if t is not None else "" for t in vocab]
        quote = tok.encode('"', add_special_tokens=False)
        bad = torch.zeros(model.get_output_embeddings().weight.shape[0], dtype=torch.bool)
        for i, s in enumerate(texts):
            if ('"' in s and s != '"') or "\n" in s or "\\" in s or "{" in s or "}" in s:
                bad[i] = True
        bad[len(texts):] = True
        for t in tok.all_special_ids:
            bad[t] = True
        _state.update(model=model, tok=tok, bad=bad.to(dev), quote=quote[0], device=dev, error=None)
    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        _state["loading"] = False
    return _state["model"]


_DANGLING = {"the", "a", "an", "in", "of", "to", "with", "and", "or", "around", "on", "at", "for", "into", "from", "by", "is", "as", "that", "while"}


def _tidy(s: str) -> str:
    s = s.strip().rstrip(",;:- ")
    words = s.split()
    while len(words) > 2 and words[-1].lower().strip(",.") in _DANGLING:
        words.pop()
    return " ".join(words).rstrip(",;:- ")


class _Decoder:
    """Greedy decoding over a growing context, with the KV cache kept between calls."""

    def __init__(self, prompt_ids: list[int]):
        self.m, self.tok, self.dev = _state["model"], _state["tok"], _state["device"]
        self.past = None
        self.logits = None
        self.deadline = time.time() + float(os.environ.get("LLM_BUDGET_S", 75))
        self.feed(prompt_ids)

    def feed(self, ids: list[int]):
        import torch
        if time.time() > self.deadline:
            raise TimeoutError("LLM too slow on this machine")
        with torch.no_grad():
            out = self.m(input_ids=torch.tensor([ids], device=self.dev), past_key_values=self.past, use_cache=True)
        self.past, self.logits = out.past_key_values, out.logits[0, -1].float()

    def text(self, s: str):
        self.feed(self.tok.encode(s, add_special_tokens=False))

    def free(self, max_tokens: int, sentence: bool = False, avoid: list[int] | None = None) -> str:
        """Generate a JSON string value up to its closing quote (or the end of the
        first sentence when sentence=True). avoid: tokens to discourage (used earlier)."""
        import torch
        got: list[int] = []
        from collections import Counter
        cnt = Counter(avoid or [])
        avoid_t = torch.tensor(list(cnt), dtype=torch.long, device=self.logits.device)
        avoid_w = torch.tensor([min(3.0, 1.5 * c) for c in cnt.values()], device=self.logits.device)
        q = _state["quote"]
        for _ in range(max_tokens):
            lg = self.logits.clone()
            lg[_state["bad"]] = -float("inf")
            if got:
                idx = torch.tensor(got, device=lg.device)
                lg[idx] = torch.where(lg[idx] > 0, lg[idx] / 1.3, lg[idx] * 1.3)  # repetition penalty
            else:
                lg[q] = -float("inf")  # no empty strings
            if len(avoid_t):
                lg[avoid_t] -= avoid_w
            t = int(torch.argmax(lg))
            if t == q:
                self.feed([t])
                return _tidy(self.tok.decode(got).strip())
            got.append(t)
            if sentence and len(got) > 3 and re.search(r"[.!?]\s*$", self.tok.decode([t])):
                self.feed([t, q])
                return self.tok.decode(got).strip()
            self.feed([t])
        self.feed([q])
        s = self.tok.decode(got).strip()
        s = re.split(r"(?<=[.!?])\s", s)[0]
        s = re.sub(r"[\s,;:-]+\S*$", "", s) if " " in s and not s.endswith((".", "!", "?")) else s  # drop a cut-off word
        return _tidy(s)

    def choose(self, options: list[str], bias: dict | None = None) -> str:
        """Pick one option (each followed by a closing quote) using a token trie.
        bias adds a log-prob bonus to an option's first token."""
        import torch
        bias = bias or {}
        seqs = {o: self.tok.encode(o + '"', add_special_tokens=False) for o in options}
        prefix: list[int] = []
        while True:
            live = [o for o, s in seqs.items() if s[:len(prefix)] == prefix]
            done = [o for o in live if len(seqs[o]) == len(prefix)]
            if done:
                return done[0]
            nxt = sorted({seqs[o][len(prefix)] for o in live})
            lp = torch.log_softmax(self.logits, -1)
            score = {t: float(lp[t]) for t in nxt}
            if not prefix:
                for o in live:
                    score[seqs[o][0]] = max(score[seqs[o][0]], float(lp[seqs[o][0]]) + bias.get(o, 0.0))
            t = max(nxt, key=lambda x: score[x])
            self.feed([t])
            prefix.append(t)


# ---------------------------------------------------------------- song sections

def sections_from_outline(outline: list[dict]) -> list[dict]:
    """Group the song's phrases into 2-4 sections of similar music energy."""
    ph = [p for p in outline if p["usable"]]
    if not ph:
        return []
    groups: list[list[dict]] = []
    for p in ph:
        if groups and groups[-1][-1]["music"] == p["music"]:
            groups[-1].append(p)
        else:
            groups.append([p])
    mean = lambda g: sum(p["level"] for p in g) / len(g)
    while len(groups) > 4:
        i = min(range(len(groups)), key=lambda k: len(groups[k]))
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if abs(mean(groups[i - 1]) - mean(groups[i])) <= abs(mean(groups[i + 1]) - mean(groups[i])) else i + 1
        a, b = sorted((i, j))
        groups[a:b + 1] = [groups[a] + groups[b]]
    if len(groups) == 1 and len(ph) >= 4:
        h = len(ph) // 2
        groups = [ph[:h], ph[h:]]
    out = []
    for k, g in enumerate(groups):
        m = mean(g)
        word = "quiet" if m < 0.34 else "loud" if m > 0.66 else "medium"
        role = ("intro" if k == 0 and word != "loud" else "ending" if k == len(groups) - 1 and len(groups) > 2 and word != "loud"
                else "peak" if word == "loud" else "build" if k < len(groups) - 1 and mean(groups[k + 1]) > m else "groove")
        out.append({"section": k + 1, "slots": [p["slot"] for p in g], "t0": g[0]["t0"], "t1": g[-1]["t1"],
                    "music": word, "role": role})
    return out


def _clock(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def mentioned_styles(text: str) -> list[str]:
    t = text.lower()
    found = []
    for code, words in STYLE_WORDS.items():
        if any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in words):
            found.append(code)
    if "JS" in found and "JB" in found and "ballet jazz" in t:
        found.remove("JS")
    if "MH" in found and "LH" in found:
        found.remove("MH")
    return found


# ---------------------------------------------------------------- planning

def rules_plan(sections: list[dict], text: str, style_hint: str = "auto", previous: dict | None = None) -> dict:
    """Fallback plan: follow the music's energy, in the style the user named."""
    styles = mentioned_styles(text) or ([style_hint] if style_hint in GENRES else [])
    t = text.lower()
    base = styles[0] if styles else (previous["sections"][0]["style"] if previous else "HO")
    mood = "calm" if re.search(r"\b(calm|chill|soft|slow|gentle|relaxed|smooth)\b", t) else \
        "intense" if re.search(r"\b(energetic|intense|hype|hard|wild|aggressive|explosive|crazy)\b", t) else None
    out = []
    targets = _targets(sections, text) if previous else None
    prev = {p["section"]: p for p in (previous or {}).get("sections", [])}
    for k, s in enumerate(sections):
        if targets is not None and s["section"] not in targets and s["section"] in prev:
            out.append({**s, **{x: prev[s["section"]][x] for x in ("style", "energy", "move", "why")}})
            continue
        style = styles[min(k, len(styles) - 1)] if styles else base
        energy = mood or {"quiet": "calm", "medium": "match", "loud": "intense"}[s["music"]]
        out.append({**s, "style": style, "energy": energy,
                    "move": {"calm": "easy groove", "match": "on-beat groove", "intense": "big hits"}[energy],
                    "why": f"The music is {s['music']} here, so the dance {'holds back' if energy == 'calm' else 'goes big' if energy == 'intense' else 'rides the beat'}."})
    if previous:
        ch = [o for o in out if targets and o["section"] in targets]
        return {"title": previous.get("title", "Your dance"), "sections": out, "parser": "rules", "changed": sorted(targets or []),
                "reply": "Rewrote " + "; ".join(f"S{o['section']} as {GENRES[o['style']]}" for o in ch) + " (quick mode)."}
    return {"title": f"{GENRES[base]} to the music", "sections": out,
            "reply": f"Here's a {GENRES[base]} routine that follows the song's energy.", "parser": "rules"}


def _prompt(sections: list[dict], text: str, previous: dict | None) -> list[dict]:
    lines = [f"S{s['section']}: {_clock(s['t0'])}-{_clock(s['t1'])}, {s['music']} music ({s['role']})" for s in sections]
    user = "Song sections:\n" + "\n".join(lines)
    if previous:
        inv = {v: k for k, v in ENERGY_WORDS.items()}
        cur = [f"S{s['section']}: {GENRES[s['style']]}, {inv.get(s['energy'], 'steady')}, {s['move']}" for s in previous["sections"]]
        user += "\n\nCurrent plan:\n" + "\n".join(cur) + f"\n\nChange request: {text[:400]}\nOnly change the sections the request is about."
    else:
        user += f"\n\nRequest: {text[:400] or 'a dance that follows the music'}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _targets(sections: list[dict], text: str) -> set[int]:
    """Which sections a change request is about (all of them if it doesn't say)."""
    t = text.lower()
    hit: set[int] = set()
    if re.search(TARGETS["last"], t):
        hit.add(sections[-1]["section"])
    if re.search(TARGETS["first"], t):
        hit.add(sections[0]["section"])
    if re.search(TARGETS["middle"], t) and len(sections) > 2:
        hit |= {s["section"] for s in sections[1:-1]}
    if re.search(TARGETS["loud"], t):
        hit |= {s["section"] for s in sections if s["music"] == "loud"} or {max(sections, key=lambda s: s["music"] == "medium")["section"]}
    if re.search(TARGETS["quiet"], t):
        hit |= {s["section"] for s in sections if s["music"] == "quiet"}
    m = re.findall(r"\bs(?:ection)?\s*(\d)\b", t)
    hit |= {int(x) for x in m if 1 <= int(x) <= len(sections)}
    return hit or {s["section"] for s in sections}


def _biases(text: str, allowed: list[str]) -> tuple[dict, dict]:
    t = text.lower()
    sb = {GENRES[c]: 3.0 for c, rx in STYLE_HINTS.items() if GENRES[c] in allowed and re.search(rx, t)}
    for c in mentioned_styles(text):
        if GENRES[c] in allowed:
            sb[GENRES[c]] = 6.0
    eb = {}
    if re.search(r"calmer|softer|gentler|slower|less", t):
        eb["calm"] = 4.0
    if re.search(r"more aggressive|harder|bigger|more energy|more intense|wilder|crazier|hype", t):
        eb["intense"] = 4.0
    return sb, eb


def plan(sections: list[dict], text: str, style_hint: str = "auto", previous: dict | None = None) -> dict:
    """Plan on the GPU worker when the Space has one, else here; rules if both fail."""
    if _remote is not None and sections and _state["model"] is not None:
        try:
            t0 = time.time()
            out = _remote(sections, text, style_hint, previous)
            out["seconds_taken"] = round(time.time() - t0, 1)
            out["device"] = "gpu"
            return out
        except Exception as e:
            _state["error"] = f"GPU call failed: {type(e).__name__}: {e}"[:300]
            return rules_plan(sections, text, style_hint, previous)
    return _plan_local(sections, text, style_hint, previous)


def _plan_local(sections: list[dict], text: str, style_hint: str = "auto", previous: dict | None = None) -> dict:
    """Return {"title", "sections": [{..., style, energy, move, why}], "reply", "parser", "seconds_taken"}."""
    t_start = time.time()
    if not sections:
        return rules_plan(sections, text, style_hint, previous)
    if _state["model"] is None:
        load()
    if _state["model"] is None:
        return rules_plan(sections, text, style_hint, previous)
    allowed = [GENRES[c] for c in GENRES]
    if not previous:
        ms = mentioned_styles(text) or ([style_hint] if style_hint in GENRES else [])
        if ms:
            allowed = [GENRES[c] for c in ms]
    style_bias, energy_bias = _biases(text, allowed)
    targets = _targets(sections, text) if previous else {s["section"] for s in sections}
    prev = {s["section"]: s for s in (previous or {}).get("sections", [])}
    inv = {v: k for k, v in ENERGY_WORDS.items()}
    try:
        with _lock:
            tok = _state["tok"]
            ids = tok.apply_chat_template(_prompt(sections, text, previous), add_generation_prompt=True)
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            d = _Decoder(list(ids))
            # Discourage copying the format example and talking about "S2" instead of the dance.
            ex = d.tok.encode(" stomps like thunder arms drifting rain storm hits hard heavy softens floats S1 S2 S3 S4 S5 S", add_special_tokens=False)
            used_move: list[int] = ex * 2
            used_why: list[int] = ex * 2
            d.text('{"title": "')
            title = d.free(10)
            title = re.split(r"[:.!,]", title)[0].strip() or title
            d.text('", "sections": [')
            out = []
            for k, s in enumerate(sections):
                d.text(("" if k == 0 else ", ") + f'{{"section": {s["section"]}, "time": "{_clock(s["t0"])}-{_clock(s["t1"])}", '
                       f'"music": "{s["music"]}", "style": "')
                keep = prev.get(s["section"]) if s["section"] not in targets else None
                if keep:  # a refinement leaves untouched sections exactly as they were
                    d.text(f'{GENRES[keep["style"]]}", "energy": "{inv.get(keep["energy"], "steady")}", "move": "{keep["move"]}", "why": "{keep["why"]}"}}')
                    out.append({**s, "style": keep["style"], "energy": keep["energy"], "move": keep["move"], "why": keep["why"]})
                    continue
                if len(allowed) == 1:
                    style = allowed[0]
                    d.text(style + '"')
                else:
                    sb = dict(style_bias)
                    if s["section"] in prev:
                        if style_bias:
                            sb = {k: v + 3.0 for k, v in sb.items()}  # the user asked for a new feel here
                        else:
                            sb[GENRES[prev[s["section"]]["style"]]] = 1.5  # otherwise lean on the old style
                    style = d.choose(allowed, sb)
                d.text(', "energy": "')
                eb = dict(energy_bias)
                if not eb:
                    eb = {"calm": 1.0} if s["music"] == "quiet" else {"intense": 1.0} if s["music"] == "loud" else {}
                    t = text.lower()
                    for e, rx in ENERGY_HINTS.items():
                        if re.search(rx, t):
                            eb[e] = eb.get(e, 0) + 0.8
                energy = d.choose(list(ENERGY_WORDS), eb)
                d.text(', "move": "')
                move = d.free(10, avoid=used_move)
                used_move += d.tok.encode(move, add_special_tokens=False)
                d.text(', "why": "')
                why = d.free(20, sentence=True, avoid=used_why)
                used_why += d.tok.encode(why, add_special_tokens=False)
                d.text("}")
                out.append({**s, "style": NAME_TO_CODE[style], "energy": ENERGY_WORDS[energy], "move": move, "why": why})
            d.text('], "reply": "')
            reply = d.free(32, sentence=True, avoid=used_why)
        if previous:  # say plainly what changed
            ch = [s for s in out if s["section"] in targets]
            reply = "Rewrote " + "; ".join(f"S{s['section']} as {GENRES[s['style']]}, {inv.get(s['energy'], 'steady')}" for s in ch) + \
                    (f". {ch[0]['why']}" if ch and ch[0]["why"] else ".")
            title = previous.get("title") or title
        elif len(reply.split()) < 5:
            reply = f"{title}: a {len(out)}-part story set to your song."
        return {"title": title or "Your dance", "sections": out, "reply": reply or "Here's your dance.",
                "parser": "llm", "seconds_taken": round(time.time() - t_start, 1),
                "changed": sorted(targets) if previous else []}
    except Exception as e:
        _state["error"] = f"generation failed: {type(e).__name__}: {e}"[:300]
        return rules_plan(sections, text, style_hint, previous)


def slots_from_plan(p: dict, n_slots: int) -> list[dict]:
    """Expand section decisions to one {style, energy} per phrase slot."""
    secs = p["sections"]
    per = [None] * n_slots
    for s in secs:
        for j in s["slots"]:
            if 0 <= j < n_slots:
                per[j] = {"style": s["style"], "energy": s["energy"], "section": s["section"]}
    for j in range(n_slots):  # pre-roll and tail slots borrow from the nearest section
        if per[j] is None:
            near = min(secs, key=lambda s: min(abs(j - x) for x in s["slots"]))
            per[j] = {"style": near["style"], "energy": near["energy"], "section": near["section"]}
    return per
