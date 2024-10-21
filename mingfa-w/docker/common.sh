#!/bin/bash
version="0.0.0"
domain="hub.byted.org"
namespace="base"
tag="$version-cuda12.5.1-devel-ubuntu20.04"
image="$domain/$namespace/x-triton:$tag"
base_image="nvcr.io/nvidia/cuda:12.5.1-devel-ubuntu20.04"
http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
#proxy_param=" --build-arg ftp_proxy=$http_proxy --build-arg http_proxy=$http_proxy --build-arg https_proxy=$http_proxy "
base_image_param=" --build-arg BASE_IMAGE=$base_image "
