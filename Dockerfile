# CPU image: song-to-dance with the AIST++ move-stitching engine and an
# on-device LLM assistant. No GPU needed.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl unzip build-essential cmake \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt
# On-device LLM for the "describe your dance" box (llama.cpp, CPU). Optional:
# if this step fails the app still works with its keyword parser.
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install --no-cache-dir llama-cpp-python==0.3.2 huggingface_hub \
 && python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'qwen2.5-0.5b-instruct-q4_k_m.gguf', local_dir='/app/models')" \
 || echo "LLM assistant not installed; using keyword parser"
COPY . .
RUN bash scripts/setup_data.sh && rm -rf data/motions data/motions.zip
ENV DATA_DIR=/tmp/jobs LLM_PATH=/app/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
WORKDIR /app/server
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
