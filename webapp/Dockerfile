FROM python:3.11-slim

WORKDIR /app

# System deps for opencv-python-headless video decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch build -- avoids pulling multi-GB CUDA wheels.
RUN pip install --no-cache-dir torch==2.5.* torchvision==0.20.* \
    --index-url https://download.pytorch.org/whl/cpu

COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install --no-cache-dir -r /app/webapp/requirements.txt

COPY nsd_inference_attention_pcr.py /app/nsd_inference_attention_pcr.py
COPY checkpoint_1600_data_attention_excluded_2to9 /app/checkpoint_1600_data_attention_excluded_2to9
COPY webapp /app/webapp

WORKDIR /app/webapp
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
