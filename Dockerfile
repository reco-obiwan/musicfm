FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ENV TERM xterm-256color

ENV WORKDIR=/opt/app
WORKDIR $WORKDIR

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && \
    apt install -y git net-tools jq vim wget htop ffmpeg python3 pip

RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc && \
    echo "export PS1='\[\e[31;1m\][\w]: \[\e[0m\]'" >> ~/.bashrc

COPY requirements.txt $WORKDIR/
RUN python3 -m pip install --upgrade pip

COPY src $WORKDIR/src
COPY res $WORKDIR/res


COPY _entrypoint.sh $WORKDIR/

ENTRYPOINT ["./_entrypoint.sh"]

CMD ["bash"]
