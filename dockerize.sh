
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
echo ">> DIR: ${DIR}"

DOCKERFILE=$DIR/Dockerfile
MODEL=${MODEL:-musicfm}
APPNAME=reco/model/$MODEL
APPTAG=${APPTAG:-latest}

IMGNAME="$APPNAME:$APPTAG"

echo "--- Docker build arguments ---"
echo "    DOCKERFILE: $DOCKERFILE"
echo "    IMGNAME: $IMGNAME"
echo "------------------------------"

docker build \
    --tag "${IMGNAME}" \
    -f "$DOCKERFILE" \
    --force-rm \
    "$DIR"
