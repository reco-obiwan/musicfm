
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
# if [ ! -d "/data01" ] ; then
#     error /data01 directory does not exists.
# fi

# if [ ! -d "/data03/music" ] ; then
#     error /data03/music directory does not exists.
# fi

# if [ ! -d "/data01/mulan_checkpoints" ] ; then
#     mkdir /data01/mulan_checkpoints
# fi

# if [ ! -d "/home/jupyterlab/work/tensorboard_log" ] ; then
#     error /home/jupyterlab/work/tensorboard_log does not exists.
# fi

if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Container ${CONTAINER_NAME} exists."
    docker stop $CONTAINER_NAME
fi

if [[ "$1" = "download" ]]; then
    wget -P ./res/ https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt
    wget -P ./res/ https://huggingface.co/minzwon/MusicFM/resolve/main/msd_stats.json
else
    docker run -it --rm --ipc=host \
        -p $PORT:8001 \
        -v /data01/aac_music:$WORKDIDR/music \
        -v /data01/musicfm:$WORKDIDR/musicfm \
        -v $DIR/src:$WORKDIDR/src \
        -v $DIR/res:$WORKDIDR/res \
        --name $CONTAINER_NAME $IMGNAME
fi

