CONTAINER_NAME=pt-yolo
IMAGE_NAME=920076894685.dkr.ecr.us-east-1.amazonaws.com/jbsnyder:pytorch-yolo

docker run -it -d --rm --gpus all --name ${CONTAINER_NAME} \
	--net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
	--ulimit=stack=67108864 --ulimit=memlock=-1 \
	-v /home/ubuntu/:/workspace/desktop \
        ${IMAGE_NAME} /bin/bash

