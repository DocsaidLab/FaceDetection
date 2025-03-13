docker run \
    --gpus all \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --net=host \
    -v $PWD:$PWD \
    -v /data:/data \
    -it --rm face_detection_dev $@