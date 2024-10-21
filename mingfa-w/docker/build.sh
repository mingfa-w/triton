#!/bin/bash -xe
. ./common.sh

DOCKER_BUILDKIT=1 docker build --progress=plain $proxy_param $base_image_param -t $image . -f Dockerfile

