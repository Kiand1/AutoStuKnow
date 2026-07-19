#!/bin/sh
set -eu

marker="/home/node/.n8n/.autostuknow-workflow-imported-v2"
workflow="/opt/autostuknow/workflows/youtube-to-rag.json"
credential_file=$(mktemp)

cleanup() {
  rm -f "$credential_file"
}
trap cleanup EXIT HUP INT TERM

if [ -z "${WEBHOOK_API_KEY:-}" ]; then
  echo "WEBHOOK_API_KEY is required; refusing to start an unprotected webhook." >&2
  exit 1
fi

node - "$credential_file" <<'NODE'
const fs = require('node:fs');
const output = process.argv[2];
const credential = [{
  id: 'AutoStuKnowWebhookHeaderAuth',
  name: 'AutoStuKnow Webhook Header Auth',
  type: 'httpHeaderAuth',
  data: {
    name: 'X-Webhook-Key',
    value: process.env.WEBHOOK_API_KEY,
  },
}];
fs.writeFileSync(output, JSON.stringify(credential), { mode: 0o600 });
NODE

echo "Synchronizing AutoStuKnow Webhook Header Auth credential..."
if ! n8n import:credentials --input="$credential_file"; then
  echo "Credential import failed; refusing to start an unprotected webhook." >&2
  exit 1
fi

if [ ! -f "$marker" ]; then
  echo "Importing and publishing the authenticated AutoStuKnow workflow..."
  if n8n import:workflow --input="$workflow" \
    && n8n publish:workflow --id=AutoStuKnowYoutubeV1; then
    touch "$marker"
  else
    echo "Authenticated workflow import failed; refusing to start." >&2
    exit 1
  fi
fi

cleanup
trap - EXIT HUP INT TERM
exec n8n start
