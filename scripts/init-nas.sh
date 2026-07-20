#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

random_secret() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    od -An -N32 -tx1 /dev/urandom | tr -d ' \n'
  fi
}

replace_placeholder() {
  key=$1
  placeholder=$2
  current=$(sed -n "s/^${key}=//p" .env | head -n 1)
  if [ "$current" = "$placeholder" ] || [ -z "$current" ]; then
    value=$(random_secret)
    if grep -q "^${key}=" .env; then
      sed -i.bak "s|^${key}=.*|${key}=${value}|" .env
    else
      printf '%s=%s\n' "$key" "$value" >> .env
    fi
    echo "Generated $key"
  fi
}

ensure_default() {
  key=$1
  value=$2
  if ! grep -q "^${key}=" .env; then
    printf '%s=%s\n' "$key" "$value" >> .env
    echo "Added $key"
  fi
}

replace_placeholder ANYTHINGLLM_JWT_SECRET replace-with-a-random-secret
replace_placeholder N8N_ENCRYPTION_KEY replace-with-a-different-random-secret
replace_placeholder INGESTOR_API_KEY replace-with-a-third-random-secret
replace_placeholder WEBHOOK_API_KEY replace-with-a-fourth-random-secret
replace_placeholder MCP_API_KEY replace-with-a-mcp-random-secret
replace_placeholder WEB_UI_PASSWORD replace-with-a-web-ui-password
replace_placeholder WEB_UI_SESSION_SECRET replace-with-a-web-ui-session-secret
ensure_default WEB_UI_USERNAME admin
ensure_default WEB_UI_SESSION_TTL_HOURS 24
ensure_default WEB_UI_SECURE_COOKIE false
ensure_default MCP_ENABLED false
ensure_default PREFER_YOUTUBE_SUBTITLES true
ensure_default ALLOW_AUTOMATIC_SUBTITLES true
rm -f .env.bak
chmod 600 .env

data_root=$(sed -n 's/^DATA_ROOT=//p' .env | head -n 1)
[ -n "$data_root" ] || data_root=./data
case "$data_root" in
  /|~|"")
    echo "Refusing unsafe DATA_ROOT: $data_root" >&2
    exit 1
    ;;
esac

mkdir -p \
  "$data_root/anythingllm" \
  "$data_root/anythingllm-hotdir" \
  "$data_root/anythingllm-outputs" \
  "$data_root/whisper" \
  "$data_root/ingestor" \
  "$data_root/n8n"

puid=$(sed -n 's/^PUID=//p' .env | head -n 1)
pgid=$(sed -n 's/^PGID=//p' .env | head -n 1)
[ -n "$puid" ] || puid=1000
[ -n "$pgid" ] || pgid=1000
if [ "$(id -u)" -eq 0 ]; then
  chown "$puid:$pgid" \
    "$data_root/anythingllm" \
    "$data_root/anythingllm-hotdir" \
    "$data_root/anythingllm-outputs" \
    "$data_root/whisper" \
    "$data_root/ingestor" \
    "$data_root/n8n"
  if [ -n "${SUDO_UID:-}" ] && [ -n "${SUDO_GID:-}" ]; then
    chown "$SUDO_UID:$SUDO_GID" .env
  fi
else
  echo "Note: if containers report Permission denied, run this script once with sudo."
fi

echo "Initialized persistent directories under $data_root"
echo "Next: edit .env, especially DATA_ROOT and N8N_PUBLIC_URL, then run docker compose up -d --build"
