"""Hugging Face Space entry point (Gradio SDK).

The Space runs `python app.py`. This prepares the dance library on first boot,
starts fetching the optional on-device LLM in the background, and serves the
FastAPI app (API + three.js page) on port 7860.
"""
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "data" / "library.npz"
MODEL_DIR = ROOT / "models"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

os.environ.setdefault("LIBRARY", str(LIBRARY))
os.environ.setdefault("LLM_PATH", str(MODEL_DIR / MODEL_FILE))
os.environ.setdefault("DATA_DIR", "/tmp/jobs")
sys.path.insert(0, str(ROOT / "server"))

# ZeroGPU hardware refuses to start unless at least one @spaces.GPU function is
# registered at import time. Nothing here needs a GPU, so this one is never called.
try:
    import spaces

    @spaces.GPU(duration=10)
    def gpu_healthcheck():
        import torch
        return torch.cuda.is_available()

    @spaces.GPU(duration=40)
    def gpu_plan(sections, text, style_hint, previous):
        """The AI choreographer's LLM on the Space's free ZeroGPU (CPU there is too slow)."""
        import choreo_llm
        return choreo_llm._plan_local(sections, text, style_hint, previous)

    ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))
    print(f"ZeroGPU: registered @spaces.GPU functions (zero_gpu={ZERO_GPU})", flush=True)
except Exception as e:  # not on ZeroGPU (local runs, Docker)
    print(f"spaces not available ({e}); running without ZeroGPU", flush=True)
    ZERO_GPU = False


def setup_gpu_llm() -> bool:
    """On ZeroGPU, load the LLM onto 'cuda' now, in the main process before launch
    (that is when ZeroGPU accepts it), and route plans through gpu_plan."""
    try:
        import choreo_llm
        choreo_llm.download()
        choreo_llm.load(device="cuda")
        if choreo_llm.model_ready():
            choreo_llm.use_remote(gpu_plan)
            print("AI choreographer LLM on ZeroGPU", flush=True)
            return True
        print(f"GPU load failed: {choreo_llm.last_error()}", flush=True)
        choreo_llm._state["error"] = None
    except Exception as e:
        print(f"GPU LLM setup failed: {e}", flush=True)
    return False


def build_library():
    if LIBRARY.exists():
        return
    import urllib.request
    import zipfile
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    zpath = data / "motions.zip"
    print("Downloading AIST++ motions...", flush=True)
    urllib.request.urlretrieve(
        "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/motions.zip", zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(data, members=[m for m in z.namelist() if not m.startswith("__MACOSX")])
    import build_library as B
    B.main(str(data / "motions"), str(LIBRARY))


def fetch_llm():
    """Download and load the AI choreographer's LLM (Qwen2.5-Instruct via transformers).
    Until it is ready, or if it fails, plans come from the rule-based fallback."""
    try:
        sys.path.insert(0, str(ROOT / "server"))
        import choreo_llm
        choreo_llm.download()
        choreo_llm.load()
        print(f"AI choreographer LLM: {choreo_llm.status()}", flush=True)
    except Exception as e:
        print(f"AI choreographer LLM unavailable, using rules: {e}", flush=True)


def load_server_app():
    import importlib.util
    spec = importlib.util.spec_from_file_location("server_app", ROOT / "server" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.app


if __name__ == "__main__":
    build_library()
    if not (ZERO_GPU and setup_gpu_llm()):
        threading.Thread(target=fetch_llm, daemon=True).start()
    import gradio as gr

    # ZeroGPU only starts a Space that launches through Gradio, so Gradio hosts the
    # page: the app itself (FastAPI + three.js) is mounted at /ui and shown full-size.
    with gr.Blocks(title="Song to Dance", css="footer{display:none!important} .gradio-container{max-width:100%!important;padding:0!important}") as demo:
        gr.HTML('<iframe src="/ui/" title="Song to Dance" style="width:100%;height:92vh;border:0;border-radius:8px"></iframe>'
                '<p style="font-size:13px;opacity:.7;margin:6px 0">Open full screen: <a href="/ui/" target="_blank">/ui/</a></p>')
    demo.queue(default_concurrency_limit=4)
    # ssr_mode=False: Spaces turn Gradio's server-side rendering on by default, and its
    # proxy would answer /ui/ with the Gradio page itself (an endless iframe loop).
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                prevent_thread_lock=True, ssr_mode=False)
    from starlette.responses import RedirectResponse
    from starlette.routing import Mount, Route
    # Ahead of Gradio's routes: the app lives at /ui, and the Space's front page
    # goes straight to it (the Gradio page only exists so ZeroGPU will start).
    demo.app.router.routes.insert(0, Mount("/ui", app=load_server_app()))
    demo.app.router.routes.insert(0, Route("/", lambda request: RedirectResponse("/ui/", status_code=307)))
    print("Song to Dance ready at /ui/", flush=True)
    demo.block_thread()
