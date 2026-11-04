#!/usr/bin/env bash
# Deploy TEKIJIN to the DGX from a self-hosted-runner checkout (#203).
#
# Runs ON the DGX (a self-hosted GitHub Actions runner, user team_a). The runner
# has checked out the develop commit into $GITHUB_WORKSPACE; this script syncs that
# tree into the live deploy dir, rebuilds the frontend, restarts the backend, and
# health-checks. On failure it restores the previous release (rollback).
#
# HARD RULES (shared GPU host):
#   * vLLM (tekijin_vllm) and Postgres (tekijin_app_pg) are NEVER touched — the GPU
#     is occupied and a restart would drop the model / the DB. Only the backend
#     (nohup uvicorn on :18000) and the frontend container are cycled.
#   * No sudo: the runner user is not root. Restart is process/docker level only.
#
# Everything is overridable via env (defaults match the DGX):
#   TEKIJIN_DEPLOY_DIR         live deploy dir           (/home/team_a/TEKIJIN)
#   TEKIJIN_VENV_PY            backend venv python       (…/tekijin-bench/.venv/bin/python)
#   TEKIJIN_API_BASE_URL       baked into the frontend   (http://internship-dgx1.tail349bcd.ts.net:18000)
#   TEKIJIN_FRONTEND_CONTAINER frontend container name   (tekijin_frontend)
#   TEKIJIN_PORT               backend port              (18000)
#   TEKIJIN_FRONTEND_PORT      frontend port             (13000)
#
# The default API URL is the host's Tailscale MagicDNS name, not its 100.x address
# (#484): Tailscale can reassign the IP, the name is tied to the hostname and does
# not move. It resolves inside the tailnet only, which is where this frontend is
# reachable anyway. NOTE: the origin the browser loads the page FROM must be listed
# in `TEKIJIN_CORS_ORIGINS` in the deployed `.env` — opening the frontend by an
# address that is not listed blocks every API call. Keep both names there while
# people still have the numeric one bookmarked.
set -euo pipefail

SRC="${GITHUB_WORKSPACE:-$(pwd)}"
DEPLOY_DIR="${TEKIJIN_DEPLOY_DIR:-/home/team_a/TEKIJIN}"
VENV_PY="${TEKIJIN_VENV_PY:-python3}"
API_BASE_URL="${TEKIJIN_API_BASE_URL:-http://internship-dgx1.tail349bcd.ts.net:18000}"
FRONTEND_CONTAINER="${TEKIJIN_FRONTEND_CONTAINER:-tekijin_frontend}"
PORT="${TEKIJIN_PORT:-18000}"
FRONTEND_PORT="${TEKIJIN_FRONTEND_PORT:-13000}"
# #529: solve capture has completed its staged rollout (#476/#519/#526/#530/#532).
# Keep the application default OFF for local/offline environments, but enable it
# on this production-specific DGX deploy path unless an operator explicitly opts out.
SLACK_SOLVE_CAPTURE_ENABLED="${TEKIJIN_SLACK_SOLVE_CAPTURE_ENABLED:-true}"
BACKUP_DIR="${DEPLOY_DIR}.prev"

# Files that live ONLY in the deployed copy (secrets, build output, installed deps)
# and must survive a --delete sync. Note: excluded paths in the destination are
# left untouched by rsync --delete, so .env / .next / node_modules are preserved.
RSYNC_EXCLUDES=(
  --exclude .git
  --exclude node_modules
  --exclude .venv
  --exclude .next
  --exclude .env
  --exclude analysis
  --exclude test-results
)

log() { echo "[deploy] $(date -u +%H:%M:%S) $*"; }

sync_tree() { # $1=from $2=to
  rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$1/" "$2/"
}

# True (0) if a requirements file differs between SRC and the live dir (or is
# missing in the live dir) — i.e. dependencies need a pip sync. cmp is quiet.
deps_changed() {
  local req
  for req in requirements.txt requirements-ml.txt; do
    if ! cmp -s "${SRC}/backend/${req}" "${DEPLOY_DIR}/backend/${req}"; then
      return 0
    fi
  done
  return 1
}

pip_sync() {
  log "requirements changed -> syncing backend deps into the venv"
  # The venv may have been created without pip (`--without-pip`), so `-m pip`
  # fails with "No module named pip" the FIRST time a requirements change triggers
  # this path (#243). Bootstrap pip idempotently before installing.
  "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$VENV_PY" -m pip install -q -r "${DEPLOY_DIR}/backend/requirements.txt"
  "$VENV_PY" -m pip install -q -r "${DEPLOY_DIR}/backend/requirements-ml.txt"
}

migrate_schema() {
  # Non-destructive schema sync (#243): `ADD COLUMN IF NOT EXISTS`, `create_all`
  # for new tables, pgvector ensure — NEVER truncates (that is `run_seed`). Without
  # this, a model change that adds a column (e.g. employees.password_hash, #241)
  # would 500 every query against that table after deploy. Idempotent, so it is
  # safe to run on every deploy. Runs BEFORE the backend restarts so the new code
  # never serves against a stale schema.
  log "apply non-destructive schema migrations"
  ( cd "${DEPLOY_DIR}/backend" && env PYTHONPATH=src "$VENV_PY" -m tekijin.data.migrate )
}

embed_missing() {
  # Fill ONLY the rows whose embedding is still NULL (embed_fixtures.py defaults to
  # only-missing), so a column a migration just added gets embedded on the next
  # deploy and every later deploy is a fast no-op (nothing missing). CPU-only +
  # offline so it never contends for vLLM's GPU or hits the network. BEST-EFFORT: a
  # failure must NOT roll back an otherwise-healthy deploy — the unembedded rows
  # simply stay out of dense search until a later run fills them. Runs after migrate
  # (the column must exist) and before the backend restart.
  log "embed rows with a NULL embedding (only-missing)"
  ( cd "${DEPLOY_DIR}/backend" \
      && env PYTHONPATH=src CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 \
         "$VENV_PY" ../scripts/embed_fixtures.py ) \
    || log "WARN embed step failed — unembedded rows stay out of dense search until next deploy; continuing"
}

build_frontend() {
  # Build INSIDE the same image the container runs, against the bind-mounted source,
  # so `next start` picks up the fresh .next on restart. NEXT_PUBLIC_* must be baked
  # at build time (it is inlined into the bundle), so it is passed here, not to the
  # running container.
  docker run --rm \
    -v "${DEPLOY_DIR}/frontend:/app" -w /app \
    -e "NEXT_PUBLIC_API_BASE_URL=${API_BASE_URL}" \
    node:20-slim bash -lc 'npm ci && npm run build'
  docker restart "$FRONTEND_CONTAINER"
}

restart_backend() {
  # Stop the previous uvicorn (best-effort; the run may not exist yet), then
  # relaunch FULLY DETACHED from this deploy job. vLLM/Postgres are separate
  # processes and are not matched.
  #
  # CRITICAL (#210): a plain `nohup … &` is NOT enough on a self-hosted runner.
  # The Actions runner tags every process it spawns with the RUNNER_TRACKING_ID
  # env var and KILLS all such processes when the job completes — so the backend
  # passed the health check and then died the instant the deploy job finished.
  # `setsid` puts it in its own session and `env -u RUNNER_TRACKING_ID` scrubs the
  # tag, so the runner's post-job cleanup no longer reaps it.
  pkill -f 'deploy/start_backend.sh' 2>/dev/null || true
  pkill -f 'uvicorn tekijin.main:app' 2>/dev/null || true
  sleep 2
  (
    cd "$DEPLOY_DIR"
    # TZ is pinned to UTC inside start_backend.sh (#456), so every launch path —
    # this one, a bare nohup, and the systemd unit — agrees without each caller
    # having to remember.
    setsid env -u RUNNER_TRACKING_ID \
      TEKIJIN_PORT="$PORT" TEKIJIN_VENV_PY="$VENV_PY" CUDA_VISIBLE_DEVICES="" \
      TEKIJIN_SLACK_SOLVE_CAPTURE_ENABLED="$SLACK_SOLVE_CAPTURE_ENABLED" \
      bash -c 'exec deploy/start_backend.sh' >"${HOME}/backend.log" 2>&1 </dev/null &
  )
}

health_check() {
  # Poll until BOTH the backend /health and the frontend root answer 200, or give up.
  local i code fcode
  for i in $(seq 1 40); do
    code=$(curl -fsS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo 000)
    fcode=$(curl -fsS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${FRONTEND_PORT}/" 2>/dev/null || echo 000)
    if [ "$code" = "200" ] && [ "$fcode" = "200" ]; then
      return 0
    fi
    sleep 3
  done
  log "health check failed (backend=${code} frontend=${fcode})"
  return 1
}

# --------------------------------------------------------------------------- #
main() {
  log "source=${SRC} -> deploy=${DEPLOY_DIR}"

  local need_pip=1
  deps_changed || need_pip=0

  log "snapshot current release -> ${BACKUP_DIR}"
  mkdir -p "$BACKUP_DIR"
  sync_tree "$DEPLOY_DIR" "$BACKUP_DIR"

  log "sync new release"
  sync_tree "$SRC" "$DEPLOY_DIR"

  if [ "$need_pip" = 1 ]; then
    pip_sync
  else
    log "requirements unchanged -> skipping pip"
  fi

  migrate_schema
  embed_missing

  log "rebuild frontend"
  build_frontend

  log "restart backend"
  restart_backend

  if health_check; then
    log "OK — backend :${PORT} and frontend :${FRONTEND_PORT} healthy"
    return 0
  fi

  log "UNHEALTHY — rolling back to the previous release"
  sync_tree "$BACKUP_DIR" "$DEPLOY_DIR"
  # The rollback tree's requirements are the previous ones; resync if they differ
  # from what is now installed is out of scope — restore source + rebuild + restart.
  build_frontend || true
  restart_backend
  if health_check; then
    log "rolled back successfully"
  else
    log "ROLLBACK ALSO UNHEALTHY — manual intervention required"
  fi
  return 1
}

main "$@"
