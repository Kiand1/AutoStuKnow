FROM denoland/deno:bin-2.8.1 AS deno

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DENO_NO_UPDATE_CHECK=1 \
    DENO_NO_PROMPT=1 \
    HOME=/home/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home app

COPY --from=deno /deno /usr/local/bin/deno

WORKDIR /app
COPY requirements.txt ./
RUN pip install --requirement requirements.txt

COPY app ./app
RUN mkdir -p /data /home/app/.cache \
    && chown -R app:app /app /data /home/app

USER app
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips=*"]
