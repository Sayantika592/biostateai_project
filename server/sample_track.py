"""A 32-second house-style demo track, synthesised in code (no samples, no licence
issues), so visitors can try the app without uploading a song.

Structure at 120 BPM (2 s per bar): 4 bars intro (soft kick, hats, chords), 4 bars groove
(kick + bass), 4 bars full (claps, open hats, stabs), 4 bars break-down and drop.
The changing intensity lets the energy matching show.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SR = 44100
BPM = 120
BEAT = 60 / BPM


def _env(n, attack=0.002, decay=0.2):
    t = np.arange(n) / SR
    a = np.clip(t / attack, 0, 1)
    return a * np.exp(-t / decay)


def _kick(n):
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)


def _noise(n, rng, decay):
    return rng.standard_normal(n) * _env(n, 0.001, decay)


def _hp(x, a=0.85):
    """One-pole high-pass, enough to make noise sound like a hat."""
    y = np.zeros_like(x)
    prev_x = prev_y = 0.0
    for i, v in enumerate(x):
        prev_y = a * (prev_y + v - prev_x)
        prev_x = v
        y[i] = prev_y
    return y


def _saw(freq, n):
    t = np.arange(n) / SR
    return 2 * ((t * freq) % 1.0) - 1


def _lp(x, k=0.08):
    y = np.zeros_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += k * (v - acc)
        y[i] = acc
    return y


def synth(seconds: float = 32.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    beats = int(seconds / BEAT)
    hat = _hp(_noise(int(0.05 * SR), rng, 0.015))
    ohat = _hp(_noise(int(0.2 * SR), rng, 0.08))
    clap = _hp(_noise(int(0.18 * SR), rng, 0.05), 0.7)
    kick = _kick(int(0.4 * SR))
    # A minor 7 -> F maj 7 -> C -> G loop, one chord per bar
    roots = [57, 53, 48, 55]
    chords = [[57, 60, 64, 67], [53, 57, 60, 64], [48, 52, 55, 60], [55, 59, 62, 65]]
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)

    def put(sig, at, gain):
        i = int(at * SR)
        j = min(n, i + len(sig))
        if i < n:
            out[i:j] += gain * sig[: j - i]

    for b in range(beats):
        t = b * BEAT
        bar = b // 4
        section = bar // 4  # 0 intro, 1 groove, 2 full, 3 break/drop
        chord = chords[bar % 4]
        if section >= 1 and not (section == 3 and bar % 4 < 2):
            put(kick, t, 0.9)
        else:
            put(kick, t, 0.4)  # soft pulse keeps the beat clear in the intro and break
        put(hat, t + BEAT / 2, 0.25 if section else 0.18)
        if section >= 2:
            put(ohat, t + BEAT / 2, 0.12)
            if b % 2 == 1:
                put(clap, t, 0.35)
        if section >= 1:
            ln = int(BEAT / 2 * SR)
            bass = _lp(_saw(hz(roots[bar % 4] - 24), ln), 0.06) * _env(ln, 0.003, 0.12)
            put(bass, t, 0.5)
        if b % 4 in (0, 2) or section >= 2:
            ln = int(0.25 * SR)
            stab = sum(_saw(hz(m), ln) for m in chord) / 4
            put(_lp(stab, 0.15) * _env(ln, 0.004, 0.09), t + (0 if section < 2 else BEAT * 0.75), 0.22)
        if section == 3 and bar % 4 == 1 and b % 4 == 3:  # riser into the drop
            ln = int(BEAT * SR)
            put(_noise(ln, rng, 10.0) * np.linspace(0, 1, ln) ** 2 * 0.4, t, 1.0)
    out /= np.abs(out).max() + 1e-9
    return (out * 0.85).astype(np.float32)


def ensure(path: Path) -> Path:
    """Write the demo track as MP3 (and keep the WAV beside it) if it isn't there yet."""
    if path.exists():
        return path
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = path.with_suffix(".wav")
    sf.write(wav, synth(), SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "160k", str(path)], check=True)
    return path
