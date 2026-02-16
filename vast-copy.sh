#\!/bin/bash
# Copy a local directory to a Vast.ai instance using SCP.
# Usage: ./vast-copy.sh <INSTANCE_ID> <LOCAL_PATH> <REMOTE_PATH>
#
# Example:
#   ./vast-copy.sh 31181236 ./remote-artifacts /workspace/remote-artifacts

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <INSTANCE_ID> <LOCAL_PATH> <REMOTE_PATH>"
    exit 1
fi

INSTANCE_ID="$1"
LOCAL_PATH="$2"
REMOTE_PATH="$3"

echo "Fetching instance details for $INSTANCE_ID..."
INSTANCE_JSON=$(uv run vastai show instance "$INSTANCE_ID" --raw)

PUBLIC_IP=$(echo "$INSTANCE_JSON" | python -c "import sys,json; print(json.load(sys.stdin)['public_ipaddr'])")
SSH_PORT=$(echo "$INSTANCE_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['ports']['22/tcp'][0]['HostPort'])")

echo "Copying $LOCAL_PATH -> $PUBLIC_IP:$SSH_PORT:$REMOTE_PATH"
scp -r -P "$SSH_PORT" -o StrictHostKeyChecking=no "$LOCAL_PATH" "root@${PUBLIC_IP}:${REMOTE_PATH}"
echo "Done."
