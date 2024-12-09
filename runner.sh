
#!/bin/bash -l

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
echo ">> DIR: ${DIR}"

DOCKERFILE=$DIR/Dockerfile
MODEL=musicfm
APPNAME=reco/model/$MODEL
APPTAG=${APPTAG:-latest}
IMGNAME="$APPNAME:$APPTAG"
CONTAINER_NAME=$MODEL
WORKDIDR=/opt/app

echo "------------------------------"
echo "    IMGNAME: $IMGNAME"
echo "------------------------------"

docker run -it --rm -d --ipc=host \
    -p $PORT:8001 \
    -v /data01/aac_music:$WORKDIDR/music \
    -v /data01/musicfm:$WORKDIDR/musicfm \
    -v $DIR/src:$WORKDIDR/src \
    -v $DIR/res:$WORKDIDR/res \
    --name $CONTAINER_NAME $IMGNAME

sleep 5

docker logs -f $CONTAINER_NAME    