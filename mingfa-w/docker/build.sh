#!/bin/bash -xe
script_dir=$(cd $(dirname $0); pwd)
cur_dir=`pwd`
cd ${script_dir}
. ./common.sh

DOCKER_BUILDKIT=1 docker build --progress=plain $proxy_param $base_image_param -t $image . -f Dockerfile
cd ${cur_dir}

