#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(dirname -- "$SCRIPT_DIR")

cd "$PROJECT_DIR"

if command -v git >/dev/null 2>&1; then
  git pull --ff-only
else
  echo "Host git not found; using a temporary alpine/git container."
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${PROJECT_DIR}:/repo" \
    -w /repo \
    alpine/git:latest pull --ff-only
fi

docker compose pull
docker compose up -d --build
docker compose ps
