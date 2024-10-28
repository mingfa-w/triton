ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.5.1-devel-ubuntu20.04
FROM ${BASE_IMAGE}

RUN echo "root:8uhb9ijn" | chpasswd
# EXPOSE 22 9405 5555

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

# install llvm for triton
ENV LLVM_INSTALL_PATH=/opt/llvm_triton-d39ee1f9-llvm-61f8a7f6
RUN mkdir /tmp/llvm && cd /tmp/llvm && \
    git clone -b mingfa-w/triton-d39ee1f9-llvm-61f8a7f6 https://github.com/mingfa-w/llvm-project.git && \
    cd llvm-project && \
    bash mingfa-w/build.sh -p ${LLVM_INSTALL_PATH} -r && \
    rm -rf /tmp/llvm

# install llvm for triton-shared
ENV LLVM_INSTALL_PATH=/opt/llvm_triton-31040564-llvm-c08c6a71
RUN mkdir /tmp/llvm && cd /tmp/llvm && \
    git clone -b mingfa-w/triton-31040564-llvm-c08c6a71 https://github.com/mingfa-w/llvm-project.git && \
    cd llvm-project && \
    bash mingfa-w/build.sh -p ${LLVM_INSTALL_PATH} -r && \
    rm -rf /tmp/llvm

# install llvm for triton-linalg
ENV LLVM_INSTALL_PATH=/opt/llvm_triton-e601be5d-llvm-10dc3a8e
RUN mkdir /tmp/llvm && cd /tmp/llvm && \
    git clone -b mingfa-w/triton-e601be5d-llvm-10dc3a8e https://github.com/mingfa-w/llvm-project.git && \
    cd llvm-project && \
    bash mingfa-w/build.sh -p ${LLVM_INSTALL_PATH} -r && \
    rm -rf /tmp/llvm

# install conda
ARG CONDA_PATH=/opt/apps/miniconda3
RUN mkdir -p ${CONDA_PATH} && cd ${CONDA_PATH} && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${CONDA_PATH}/miniconda.sh && \
    bash ${CONDA_PATH}/miniconda.sh -b -u -p ${CONDA_PATH} && \
    rm ${CONDA_PATH}/miniconda.sh && \
    ${CONDA_PATH}/bin/conda create -n triton-py3124 python=3.12.4

# create python venv for py3-triton
ARG PY3_TRITON=/opt/apps/py3-triton
RUN ${CONDA_PATH}/envs/triton-py3124/bin/python -m venv ${PY3_TRITON} && \
    ${PY3_TRITON}/bin/pip install --upgrade pip && \
    ${PY3_TRITON}/bin/pip3 install wheel setuptools ninja wheel pybind11 torch numpy scipy pytest lit pandas matplotlib && \
    chmod 777 -R ${PY3_TRITON}

# build triton
ENV PATH=${PY3_TRITON}/bin:$PATH
ENV PATH=${PY3_TRITON}/bin/activate:$PATH
RUN . ${PY3_TRITON}/bin/activate

ARG TRITON=/opt/apps/triton-src
RUN mkdir -p ${TRITON} && cd ${TRITON} && \
    git clone -b mingfa-w/d39ee1f9 https://github.com/mingfa-w/triton.git && \
    cd triton && \
    bash mingfa-w/build.sh

# create python venv for py3-triton-shared
ARG PY3_TRITON_SHARED=/opt/apps/py3-triton-shared
RUN ${CONDA_PATH}/envs/triton-py3124/bin/python -m venv ${PY3_TRITON_SHARED} && \
    ${PY3_TRITON_SHARED}/bin/pip install --upgrade pip && \
    ${PY3_TRITON_SHARED}/bin/pip3 install wheel setuptools ninja wheel pybind11 torch numpy scipy pytest lit pandas matplotlib && \
    chmod 777 -R ${PY3_TRITON_SHARED}

# build triton-shared
ENV PATH=${PY3_TRITON_SHARED}/bin:$PATH
ENV PATH=${PY3_TRITON_SHARED}/bin/activate:$PATH
RUN . ${PY3_TRITON_SHARED}/bin/activate

ARG TRITON=/opt/apps/triton-shared-src
RUN mkdir -p ${TRITON} && cd ${TRITON} && \
    git clone --recurse-submodules -b mingfa-w/5bd61a02 https://github.com/mingfa-w/triton-shared.git && \
    cd triton-shared && \
    bash mingfa-w/build.sh

# create python venv for py3-triton-linalg
ARG PY3_TRITON_LINALG=/opt/apps/py3-triton-linalg
RUN ${CONDA_PATH}/envs/triton-py3124/bin/python -m venv ${PY3_TRITON_LINALG} && \
    ${PY3_TRITON_LINALG}/bin/pip install --upgrade pip && \
    ${PY3_TRITON_LINALG}/bin/pip3 install wheel setuptools ninja wheel pybind11 torch numpy scipy pytest lit pandas matplotlib && \
    chmod 777 -R ${PY3_TRITON_LINALG}

# build triton-linalg
ENV PATH=${PY3_TRITON_LINALG}/bin:$PATH
ENV PATH=${PY3_TRITON_LINALG}/bin/activate:$PATH
RUN . ${PY3_TRITON_LINALG}/bin/activate

ARG TRITON=/opt/apps/triton-linalg-src
RUN mkdir -p ${TRITON} && cd ${TRITON} && \
    git clone --recurse-submodules -b mingfa-w/e601be5d https://github.com/mingfa-w/triton-linalg.git && \
    cd triton-linalg && \
    bash mingfa-w/build.sh

# 设置SSH无密码登录（如果需要）
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    mkdir -p /var/run/sshd

EXPOSE 22 9000 9001 9002 9003 9004 9005 9006
# 设置sshd自启动
CMD ["/usr/sbin/sshd", "-D"]
# CMD ["source", "/opt/apps/py3-triton/bin/activate"]
WORKDIR /host
# ENTRYPOINT ["/bin/bash"]
# ENTRYPOINT ["source", "/opt/apps/py3-triton/bin/activate"]
 
