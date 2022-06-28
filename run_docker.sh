#!/bin/bash

DLC_ACCOUNT=763104351884
REGION=us-east-1
AWS_ACCOUNT=`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`

REPO=jbsnyder
TAG=pytorch-yolo
CONTAINER_NAME=pt_dlc
IMAGE_NAME=${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}

docker run -it --rm --gpus all --name ${CONTAINER_NAME} \
            --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
            --ulimit=stack=67108864 --ulimit=memlock=-1 \
            -v /home/ec2-user/SageMaker/data/:/opt/ml/input/data/ \
            -v /home/ec2-user/SageMaker/ScaledYOLOv4/:/opt/ml/code/ \
            ${IMAGE_NAME} /bin/bash -c /opt/ml/code/run.sh
