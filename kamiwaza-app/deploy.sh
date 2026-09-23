#!/usr/bin/env bash
# Build and deploy the SC2 Command Console to a Kamiwaza instance as an App
# Garden extension. Follows the flow proven on srt.gaifederal.net for
# ai-tutor: image pushed to the node-local registry (localhost:5001, where the
# extension operator rewrites external image references anyway), extension
# created through the API, catalog template kept in step with every release.
#
# Usage (from this directory, on the Kamiwaza node itself):
#   TAG=0.1.0 ./deploy.sh build       # docker build + push
#   TAG=0.1.0 ./deploy.sh create      # first deploy: create the extension
#   TAG=0.1.1 ./deploy.sh patch       # later releases: move the running app to TAG
#   TAG=0.1.1 ./deploy.sh template    # create/update the App Garden catalog entry
#   ./deploy.sh model                 # print the in-mesh LLM URL it would use
#
# Env: KZ_HOST (default https://srt.gaifederal.net), KAMIWAZA_USER (default
# admin), KAMIWAZA_PASS (prompted for if unset), MODEL_NAME (default
# gpt-oss-120b), APP_NAME (default sc2-command).
#
# The pod holds no platform credential. Browser and game requests are
# authenticated by the platform ingress; the only outbound call the app makes
# is to the in-mesh vLLM service, which needs no auth.
set -euo pipefail
cd "$(dirname "$0")"

KZ_HOST="${KZ_HOST:-https://srt.gaifederal.net}"
API="$KZ_HOST/api"
APP_NAME="${APP_NAME:-sc2-command}"
MODEL_NAME="${MODEL_NAME:-gpt-oss-120b}"
IMAGE_REPO="localhost:5001/jamesimaher/sc2-command"
PLATFORM_NS="${PLATFORM_NS:-kamiwaza}"

need_tag() { TAG="${TAG:?set TAG, e.g. TAG=0.1.0 -- use a new tag for every push (IfNotPresent pull policy)}"; }

token() {
  if [ -z "${KAMIWAZA_PASS:-}" ]; then
    read -r -s -p "Kamiwaza password for ${KAMIWAZA_USER:-admin}: " KAMIWAZA_PASS; echo >&2
  fi
  curl -sk -X POST "$API/auth/token" -H "Content-Type: application/x-www-form-urlencoded" \
    --data-urlencode "username=${KAMIWAZA_USER:-admin}" --data-urlencode "password=${KAMIWAZA_PASS}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])"
}

# In-mesh vLLM Service for the DEPLOYED model: vllm-<first 8 of deployment id>.
# The public /runtime/models/<id> route doesn't resolve from inside pods.
model_url() {
  curl -sk "$API/serving/deployments" -H "Authorization: Bearer $TOK" | python3 -c "
import sys, json
rows = [d for d in json.load(sys.stdin) if d.get('status') == 'DEPLOYED' and d.get('m_name') == '$MODEL_NAME']
if not rows: sys.exit('no DEPLOYED deployment of $MODEL_NAME')
print(f\"http://vllm-{rows[0]['id'][:8]}.$PLATFORM_NS.svc.cluster.local:8000/v1\")
"
}

running_name() {
  curl -sk -H "Authorization: Bearer $TOK" "$API/extensions" | python3 -c "
import sys, json
names = [e['name'] for e in json.load(sys.stdin)
         if e.get('type') == 'app' and (e['name'] == '$APP_NAME' or e['name'].startswith('$APP_NAME-'))]
print('\n'.join(names))
"
}

case "${1:-}" in
  build)
    need_tag
    docker build -t "$IMAGE_REPO:$TAG" .
    docker push "$IMAGE_REPO:$TAG"
    ;;

  model)
    TOK="$(token)"; model_url
    ;;

  create)
    need_tag; TOK="$(token)"
    if [ -n "$(running_name)" ]; then
      echo "$APP_NAME is already running as: $(running_name) -- use: TAG=$TAG ./deploy.sh patch" >&2; exit 1
    fi
    export TAG LLM_URL="$(model_url)" APP_NAME MODEL_NAME IMAGE_REPO KZ_HOST PLATFORM_NS
    payload="$(mktemp)"; trap 'rm -f "$payload"' EXIT
    python3 - "$payload" <<'PY'
import json, os, sys
e = os.environ
json.dump({
    "name": e["APP_NAME"], "type": "app", "version": e["TAG"],
    "services": [{
        "name": "app",
        "image": f"{e['IMAGE_REPO']}:{e['TAG']}",
        "primary": True,
        "ports": [{"container_port": 8000, "protocol": "TCP"}],
        "env": [
            {"name": "KAMIWAZA_BASE_URL", "value": e["LLM_URL"]},
            {"name": "MODEL_NAME", "value": e["MODEL_NAME"]},
        ],
        "replicas": 1,  # relay state is in memory
        "healthCheck": {"httpGet": {"path": "/health", "port": 8000}, "initialDelaySeconds": 5, "periodSeconds": 15},
    }],
    # All four are needed for the App Garden dashboard's "Primary Route" to populate.
    "kamiwaza": {
        "use_auth": "true",
        "api_url": f"http://core-api.{e['PLATFORM_NS']}.svc.cluster.local:7777/api",
        "public_api_url": f"{e['KZ_HOST']}/api",
        "origin": e["KZ_HOST"],
    },
    "networking": {"ingress_enabled": True, "path_prefix": f"/runtime/apps/{e['APP_NAME']}"},
}, open(sys.argv[1], "w"))
PY
    curl -sk -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
      "$API/extensions" --data @"$payload" -w "\n[HTTP %{http_code}]\n"
    echo "App URL: $KZ_HOST/runtime/apps/$APP_NAME/"
    ;;

  patch)
    need_tag; TOK="$(token)"
    NAMES="$(running_name)"
    [ -n "$NAMES" ] || { echo "No running $APP_NAME -- use: TAG=$TAG ./deploy.sh create" >&2; exit 1; }
    [ "$(echo "$NAMES" | wc -l)" -eq 1 ] || { echo "More than one instance running:" >&2; echo "$NAMES" >&2; exit 1; }
    curl -sk -o /dev/null -w "PATCH $NAMES -> $TAG: HTTP %{http_code}\n" -X PATCH \
      -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
      "$API/extensions/$NAMES" -d "{\"services\":[{\"name\":\"app\",\"image\":{\"tag\":\"$TAG\"}}]}"
    echo "App URL: $KZ_HOST/runtime/apps/$NAMES/"
    ;;

  template)
    # What the App Garden catalog shows and what a stop/restart in the UI
    # launches. Deploys and patches don't touch it, so run this every release.
    need_tag; TOK="$(token)"
    export TAG LLM_URL="$(model_url)" APP_NAME MODEL_NAME IMAGE_REPO
    payload="$(mktemp)"; trap 'rm -f "$payload"' EXIT
    python3 - "$payload" <<'PY'
import json, os, sys
e = os.environ
compose = f"""services:
  app:
    image: {e['IMAGE_REPO']}:{e['TAG']}
    ports:
      - "8000"
    environment:
      - KAMIWAZA_BASE_URL=${{KAMIWAZA_BASE_URL:-{e['LLM_URL']}}}
      - MODEL_NAME=${{MODEL_NAME:-{e['MODEL_NAME']}}}
      - KAMIWAZA_USE_AUTH=${{KAMIWAZA_USE_AUTH:-true}}
"""
json.dump({
    "name": e["APP_NAME"], "version": e["TAG"], "source_type": "user_repo", "visibility": "private",
    "template_type": "app", "risk_tier": 1,
    "description": json.load(open("kamiwaza.json"))["description"],
    "compose_yml": compose,
    "env_defaults": {"KAMIWAZA_BASE_URL": e["LLM_URL"], "MODEL_NAME": e["MODEL_NAME"], "KAMIWAZA_USE_AUTH": "true"},
}, open(sys.argv[1], "w"))
PY
    TEMPLATE_ID="$(curl -sk -H "Authorization: Bearer $TOK" "$API/apps/app_templates" \
      | python3 -c "import sys,json; print(next((t['id'] for t in json.load(sys.stdin) if t['name']=='$APP_NAME'), ''))")"
    if [ -n "$TEMPLATE_ID" ]; then
      curl -sk -X PUT -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
        "$API/apps/app_templates/$TEMPLATE_ID" --data @"$payload" -w "\n[HTTP %{http_code}]\n"
    else
      curl -sk -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
        "$API/apps/app_templates" --data @"$payload" -w "\n[HTTP %{http_code}]\n"
    fi
    ;;

  *)
    sed -n '2,20p' "$0"; exit 1
    ;;
esac
