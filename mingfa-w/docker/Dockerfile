ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.5.1-devel-ubuntu20.04
FROM ${BASE_IMAGE}

RUN echo "root:8uhb9ijn" | chpasswd
EXPOSE 22 9405 5555

RUN apt-get update && apt-get -y --no-install-recommends install \
    wget \
    git \
    clang lld ninja-build ccache libssl-dev \
    vim \
    build-essential \
    python3-dev python3-pip python3-venv \
    sudo \
    iputils-ping \
    net-tools \
    openssh-client \
    openssh-server \
    gdb \
    > /dev/null && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists

# install cmake
RUN mkdir /tmp/cmake && cd /tmp/cmake && \
	wget https://cmake.org/files/v3.20/cmake-3.20.0.tar.gz && \
	tar -zxvf cmake-3.20.0.tar.gz && \
	cd cmake-3.20.0 && \
    ./configure && make -j$(nproc) && make install && \
    rm -rf /tmp/cmake

# install llvm
ENV LLVM_INSTALL_PATH=/opt/llvm_triton-d39ee1f9-llvm-61f8a7f6
RUN mkdir /tmp/llvm && cd /tmp/llvm && \
    git clone -b mingfa-w/triton-d39ee1f9-llvm-61f8a7f6 https://github.com/mingfa-w/llvm-project.git && \
    cd llvm-project && \
    bash mingfa-w/build.sh -p ${LLVM_INSTALL_PATH} -r && \
    rm -rf /tmp/llvm
# ENV PATH=${LLVM_INSTALL_PATH}/bin:$PATH

# create python venv
ARG PY3_TRITON=/opt/apps/py3-triton
RUN python3 -m venv ${PY3_TRITON} && \
    ${PY3_TRITON}/bin/pip3 install wheel setuptools ninja wheel pybind11 torch numpy scipy pytest lit pandas matplotlib && \
    chmod 777 -R ${PY3_TRITON}

EXPOSE 22 9000 9001 9002 9003 9004 9005 9006   
WORKDIR /host
# ENTRYPOINT "/bin/bash"
 
