"""Song-to-dance API.

Upload a song, get a generated SMPL dance back as viewer JSON or BVH.

Engines (MODE env var, picked automatically if unset):
  stitch  beat-matched retrieval and stitching of real AIST++ moves (CPU, default
          when LIBRARY points at a built library.npz)
  edge    pretrained EDGE diffusion model (GPU, when EDGE_DIR has checkpoint.pt)
  mock    procedural motion, for UI work only
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import motion

EDGE_DIR = Path(os.environ.get("EDGE_DIR", "/opt/EDGE"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/choreo-jobs"))
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
LIBRARY = Path(os.environ.get("LIBRARY", Path(__file__).resolve().parent.parent / "data" / "library.npz"))
MODE = os.environ.get("MODE") or (
    "stitch" if LIBRARY.exists() else "edge" if (EDGE_DIR / "checkpoint.pt").exists() else "mock")
MAX_SECONDS = 30 if MODE == "edge" else 180
_choreo = None


def choreographer():
    global _choreo
    if _choreo is None:
        import choreographer as C
        _choreo = C.Choreographer(C.Library.load(LIBRARY))
    return _choreo
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Choreography API")
app.add_middleware(GZipMiddleware, minimum_size=2048)  # motion JSON shrinks ~4x
jobs: dict[str, dict] = {}
job_locks: dict[str, threading.Lock] = {}
gpu_lock = threading.Lock()  # one EDGE run at a time per GPU


def _ffmpeg_clip(src: Path, dst: Path, start: float, seconds: float):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(seconds),
         "-i", str(src), "-ac", "1", "-ar", "44100", str(dst)],
        check=True,
    )
    # Compressed copy for the browser: ~5x smaller than the WAV, so playback never stalls.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(dst), "-b:a", "128k", str(dst.with_suffix(".mp3"))],
        check=True,
    )


def _write_outputs(job_dir: Path, sample: dict, info: dict | None = None):
    view = motion.to_viewer_json(sample)
    if info:
        view["beats"] = info.get("beats", [])
        view["timeline"] = info.get("timeline", [])
    (job_dir / "motion.json").write_text(json.dumps(view, separators=(",", ":")))
    (job_dir / "motion.bvh").write_text(motion.to_bvh(sample))
    with open(job_dir / "motion.pkl", "wb") as f:
        pickle.dump(sample, f)


def _run_edge(job_dir: Path, clip: Path, seconds: int) -> dict:
    # EDGE slices the song into 5 s windows (2.5 s stride) and picks a random
    # contiguous run of out_length seconds. The clip is trimmed to exactly
    # out_length, so the only possible run starts at 0 and stays in sync.
    music = job_dir / "music"
    music.mkdir(exist_ok=True)
    shutil.copy(clip, music / "song.wav")
    out = job_dir / "motions"
    cmd = [
        sys.executable, "test.py",
        "--feature_type", "jukebox",
        "--out_length", str(seconds),
        "--music_dir", str(music),
        "--checkpoint", str(EDGE_DIR / "checkpoint.pt"),
        "--save_motions", "--motion_save_dir", str(out),
        "--render_dir", str(job_dir / "renders"),
        "--no_render",
    ]
    with gpu_lock:
        proc = subprocess.run(cmd, cwd=EDGE_DIR, capture_output=True, text=True)
    (job_dir / "edge.log").write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"EDGE failed: {proc.stderr[-800:]}")
    pkls = glob.glob(str(out / "**" / "*.pkl"), recursive=True)
    if not pkls:
        raise RuntimeError("EDGE produced no motion file")
    with open(pkls[0], "rb") as f:
        return pickle.load(f)


def _plan_dance(job: dict, clip: Path, seconds: int, genre: str, prompt: str, previous: dict | None = None):
    """The AI choreographer writes (or rewrites) the plan; returns per-slot {style, energy}."""
    import choreographer as C
    import choreo_llm as L
    outline = C.song_outline(clip, seconds)
    sections = L.sections_from_outline(outline)
    import constraints as K
    new_cons = K.parse(prompt, sections)
    body_only = previous and (new_cons or K.clears(prompt)) and not L.mentioned_styles(prompt) and \
        not re.search(r"calm|intense|energ|aggress|wild|soft|hard|faster|slower|style|bigger|graceful", prompt.lower())
    if body_only:  # "keep the left hand still": same plan, new body rule, no LLM call needed
        plan = {**previous, "parser": previous.get("parser", "rules"), "changed": [],
                "reply": "Done: " + ("removed the body constraints." if K.clears(prompt) else "; ".join(c["label"] for c in new_cons) + ".")}
    elif prompt.strip() or previous:
        plan = L.plan(sections, prompt, style_hint=genre, previous=previous)
    else:
        plan = L.rules_plan(sections, prompt, style_hint=genre if genre in C.GENRES else "auto")
    if K.clears(prompt):
        job["constraints"] = []
    else:
        job["constraints"] = K.merge(job.get("constraints", []) if previous else [], new_cons)
    plan["constraints"] = job["constraints"]
    job["plan"] = plan
    job["plan_slots"] = L.slots_from_plan(plan, len(outline) + 2)
    return job["plan_slots"]


def _annotate(job: dict, info: dict):
    """Attach the plan's section, move idea and reason to each timeline card."""
    secs = {s["section"]: s for s in job.get("plan", {}).get("sections", [])}
    slots = job.get("plan_slots") or []
    for t in info.get("timeline", []):
        ps = slots[t["slot"]] if t["slot"] < len(slots) else None
        sec = secs.get(ps["section"]) if ps else None
        if sec:
            t.update(section=sec["section"], move=sec["move"], plan_energy=sec["energy"])


def _work(job_id: str, src: Path, start: float, seconds: int, genre: str, seed: int, energy: str, prompt: str = ""):
    job = jobs[job_id]
    job_dir = DATA_DIR / job_id
    try:
        clip = job_dir / "clip.wav"
        job["status"] = "preparing audio"
        _ffmpeg_clip(src, clip, start, seconds)
        info = None
        if MODE == "stitch":
            job["status"] = "AI choreographer is writing your plan" if prompt.strip() else "reading the music"
            slots = _plan_dance(job, clip, seconds, genre, prompt)
            if energy != "match" and not prompt.strip():
                slots = [{**x, "energy": energy} for x in slots]
                job["plan_slots"] = slots
            job["status"] = "dancing it out"
            sample, info = choreographer().generate(clip, seconds, genre=genre, seed=seed, energy=energy, plan=slots,
                                                    constraints=job.get("constraints"))
            _annotate(job, info)
            job.update({k: v for k, v in info.items() if k not in ("beats",)})
        elif MODE == "edge":
            job["status"] = "generating dance"
            sample = _run_edge(job_dir, clip, seconds)
        else:
            sample = motion.mock_sample(seconds, seed=int(uuid.UUID(job_id)) % 2**32)
        _write_outputs(job_dir, sample, info)
        job["version"] = job.get("version", 0) + 1
        job["status"] = "done"
    except Exception as e:  # surfaced to the client
        job["status"] = "error"
        job["error"] = str(e)


@app.get("/api/health")
def health():
    import choreographer as C
    import choreo_llm
    return {"mode": MODE, "max_seconds": MAX_SECONDS, "genres": C.GENRES if MODE == "stitch" else {},
            "assistant": choreo_llm.status(), "llm": choreo_llm.LLM_REPO, "llm_error": choreo_llm.last_error()}


@app.post("/api/generate")
async def generate(file: UploadFile = File(...), start: float = Form(0.0), seconds: int = Form(30),
                   genre: str = Form("auto"), seed: int | None = Form(None), energy: str = Form("match"),
                   prompt: str = Form("")):
    # EDGE's window maths needs a multiple of 5 s, at least 5 s.
    seconds = max(5, min(MAX_SECONDS, int(round(seconds / 5)) * 5))
    seed = seed if seed is not None else int.from_bytes(os.urandom(4), "little")
    job_id = str(uuid.uuid4())
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True)
    src = job_dir / ("upload" + Path(file.filename or "song").suffix)
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    jobs[job_id] = {"id": job_id, "status": "queued", "seconds": seconds, "start": max(0.0, start)}
    threading.Thread(target=_work, args=(job_id, src, max(0.0, start), seconds, genre, seed, energy if energy in ("calm", "match", "intense") else "match", prompt[:500]), daemon=True).start()
    return jobs[job_id]


class Prompt(BaseModel):
    text: str


@app.post("/api/interpret")
def interpret(p: Prompt):
    """Chat box: turn a typed request into dance settings with the on-device LLM."""
    import intent
    return intent.interpret(p.text)


class Reroll(BaseModel):
    slot: int


@app.post("/api/jobs/{job_id}/reroll")
def reroll(job_id: str, r: Reroll):
    """Swap one 8-beat phrase for a different move, keeping every other phrase."""
    job = jobs.get(job_id)
    if not job or job.get("status") != "done" or "picks" not in job:
        raise HTTPException(409, "dance not ready")
    picks = job["picks"]
    if not 0 <= r.slot < len(picks):
        raise HTTPException(400, "no such phrase")
    lock = job_locks.setdefault(job_id, threading.Lock())
    with lock:
        job_dir = DATA_DIR / job_id
        banned = set(job.setdefault("banned", {}).get(str(r.slot), [])) | {picks[r.slot]}
        fixed = {j: pid for j, pid in enumerate(picks) if j != r.slot}
        sample, info = choreographer().generate(
            job_dir / "clip.wav", job["seconds"], genre=job["genre_code"], energy=job.get("energy", "match"),
            seed=int.from_bytes(os.urandom(4), "little"), fixed=fixed, exclude={r.slot: banned}, plan=job.get("plan_slots"),
            constraints=job.get("constraints"))
        job["banned"][str(r.slot)] = sorted(banned)[-12:]
        _annotate(job, info)
        job.update({k: v for k, v in info.items() if k not in ("beats",)})
        _write_outputs(job_dir, sample, info)
        job["version"] = job.get("version", 0) + 1
    return job


class Refine(BaseModel):
    text: str


@app.post("/api/jobs/{job_id}/refine")
def refine(job_id: str, r: Refine):
    """Chat with the choreographer: rewrite the plan in words, keep the phrases it didn't touch."""
    job = jobs.get(job_id)
    if not job or job.get("status") != "done" or "plan" not in job:
        raise HTTPException(409, "dance not ready")
    lock = job_locks.setdefault(job_id, threading.Lock())
    with lock:
        job_dir = DATA_DIR / job_id
        old_slots, picks = job["plan_slots"], job["picks"]
        slots = _plan_dance(job, job_dir / "clip.wav", job["seconds"], job["genre_code"], r.text[:500], previous=job["plan"])
        fixed = {j: pid for j, pid in enumerate(picks)
                 if j < len(slots) and j < len(old_slots) and (slots[j]["style"], slots[j]["energy"]) == (old_slots[j]["style"], old_slots[j]["energy"])}
        sample, info = choreographer().generate(
            job_dir / "clip.wav", job["seconds"], genre=job["genre_code"], seed=int.from_bytes(os.urandom(4), "little"),
            fixed=fixed, plan=slots, constraints=job.get("constraints"))
        _annotate(job, info)
        job.update({k: v for k, v in info.items() if k not in ("beats",)})
        job["banned"] = {}
        _write_outputs(job_dir, sample, info)
        job["version"] = job.get("version", 0) + 1
        job["changed_slots"] = [j for j in range(len(picks)) if j not in fixed]
    return job


_sample_lock = threading.Lock()


@app.get("/api/sample.mp3")
def sample_song():
    """Built-in demo track, synthesised on first request."""
    import sample_track
    with _sample_lock:
        path = sample_track.ensure(DATA_DIR / "sample" / "sample.mp3")
    return FileResponse(path, media_type="audio/mpeg", filename="song-to-dance-demo.mp3")


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    return jobs[job_id]


FILES = {"motion.json": "application/json", "motion.bvh": "text/plain", "clip.wav": "audio/wav",
         "clip.mp3": "audio/mpeg", "motion.pkl": "application/octet-stream"}


@app.get("/api/jobs/{job_id}/{name}")
def job_file(job_id: str, name: str):
    if job_id not in jobs or name not in FILES:
        raise HTTPException(404)
    path = DATA_DIR / job_id / name
    if not path.exists():
        raise HTTPException(409, "not ready")
    headers = {"Cache-Control": "no-store"}
    return FileResponse(path, media_type=FILES[name], filename=f"dance-{job_id[:8]}{Path(name).suffix}", headers=headers)


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
