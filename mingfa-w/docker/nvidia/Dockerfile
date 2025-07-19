ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.5.1-devel-ubuntu20.04
ARG ssh_prv_key
ARG ssh_pub_key
FROM ${BASE_IMAGE}

RUN echo "root:8uhb9ijn" | chpasswd
EXPOSE 22

ARG APPS_PATH=/opt/apps
ARG CODE_PATH=${APPS_PATH}/code
ARG VENV_PATH=${APPS_PATH}/venv
ARG http_proxy
ARG ftp_proxy
ARG https_proxy
ENV http_proxy=${http_proxy}
ENV https_proxy=${http_proxy}
ENV ftp_proxy=${http_proxy}

RUN apt-get update && apt-get -y --no-install-recommends install \
    wget \
    git \
    clang lld ninja-build ccache libssl-dev clang-format \
    vim \
    build-essential \
    python3-dev python3-pip python3-venv \
    sudo \
    iputils-ping \
    net-tools \
    openssh-client \
    openssh-server \
    gdb \
    tree \
    > /dev/null && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists

# == 配置ssh private key ==
# Make ssh dir
RUN mkdir /root/.ssh/
# Copy over private key, and set permissions
ADD id_rsa /root/.ssh/id_rsa
ADD id_rsa.pub /root/.ssh/id_rsa.pub
RUN chmod 700 /root/.ssh/id_rsa && \
    chmod 700 /root/.ssh/id_rsa.pub && \
    chown -R root:root /root/.ssh && \
    echo "Host code.byted.org\n\tStrictHostKeyChecking no\n" >> /root/.ssh/config && \
    git config --global url.ssh://git@code.byted.org/.insteadOf https://code.byted.org/ && \
    touch /root/.ssh/known_hosts && \
    # 设置SSH无密码登录（如果需要）
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    mkdir -p /var/run/sshd

# install cmake
RUN mkdir /tmp/cmake && cd /tmp/cmake && \
	wget https://cmake.org/files/v3.20/cmake-3.20.0.tar.gz && \
	tar -zxvf cmake-3.20.0.tar.gz && \
	cd cmake-3.20.0 && \
    ./configure && make -j$(nproc) && make install && \
    rm -rf /tmp/cmake

# install llvm for triton-x
ARG LLVM_INSTALL_PATH=${APPS_PATH}/llvm_b5cc222
RUN mkdir /tmp/llvm && cd /tmp/llvm && \
    git clone -b mingfa-w/b5cc222 https://github.com/mingfa-w/llvm-project.git && \
    cd llvm-project && \
    bash mingfa-w/build.sh -p ${LLVM_INSTALL_PATH} -r && \
    rm -rf /tmp/llvm

# install llvm for triton-linalg
ARG LLVM_INSTALL_PATH=${APPS_PATH}/llvm_10dc3a8
RUN mkdir /tmp/llvm && cd /tmp/llvm && \
    git clone -b mingfa-w/10dc3a8 https://github.com/mingfa-w/llvm-project.git && \
    cd llvm-project && \
    bash mingfa-w/build.sh -p ${LLVM_INSTALL_PATH} -r && \
    rm -rf /tmp/llvm

# install llvm for triton-shared
ARG LLVM_INSTALL_PATH=${APPS_PATH}/llvm_c08c6a7
RUN mkdir /tmp/llvm && cd /tmp/llvm && \
    git clone -b mingfa-w/c08c6a7 https://github.com/mingfa-w/llvm-project.git && \
    cd llvm-project && \
    bash mingfa-w/build.sh -p ${LLVM_INSTALL_PATH} -r && \
    rm -rf /tmp/llvm

# install conda, 并创建一个python 3.12.4的虚拟环境
ARG CONDA_PATH=${APPS_PATH}/miniconda3
RUN mkdir -p ${CONDA_PATH} && cd ${CONDA_PATH} && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${CONDA_PATH}/miniconda.sh && \
    bash ${CONDA_PATH}/miniconda.sh -b -u -p ${CONDA_PATH} && \
    rm ${CONDA_PATH}/miniconda.sh && \
    ${CONDA_PATH}/bin/conda create -n triton-py3124 python=3.12.4

# create python venv for triton-x
ARG PY3_TRITON_X=${VENV_PATH}/triton-x
RUN ${CONDA_PATH}/envs/triton-py3124/bin/python -m venv ${PY3_TRITON_X} && \
    ${PY3_TRITON_X}/bin/pip install --upgrade pip && \
    ${PY3_TRITON_X}/bin/pip3 install wheel setuptools ninja wheel pybind11 torch numpy scipy pytest lit pandas matplotlib && \
    chmod 777 -R ${PY3_TRITON_X}

# # build triton-x
# ENV PATH=${PY3_TRITON_X}/bin:$PATH
# ENV PATH=${PY3_TRITON_X}/bin/activate:$PATH
# RUN . ${PY3_TRITON_X}/bin/activate
# RUN mkdir -p ${CODE_PATH} && cd ${CODE_PATH} && \
#     git clone --recurse-submodules -b master git@code.byted.org:seed/triton-x.git && \
#     cd triton-x && \
#     bash utils/script/build-triton-x.sh

# create python venv for py3-triton-shared
ARG PY3_TRITON_SHARED=${VENV_PATH}/triton-shared
RUN ${CONDA_PATH}/envs/triton-py3124/bin/python -m venv ${PY3_TRITON_SHARED} && \
    ${PY3_TRITON_SHARED}/bin/pip install --upgrade pip && \
    ${PY3_TRITON_SHARED}/bin/pip3 install wheel setuptools ninja wheel pybind11 torch numpy scipy pytest lit pandas matplotlib && \
    chmod 777 -R ${PY3_TRITON_SHARED}

# build triton-shared
# RUN . ${PY3_TRITON_SHARED}/bin/activate && \
#     mkdir -p ${CODE_PATH} && cd ${CODE_PATH} && \
#     git clone --recurse-submodules -b mingfa-w/5bd61a02 https://github.com/mingfa-w/triton-shared.git && \
#     cd triton-shared && \
#     which python && python --version && bash mingfa-w/build.sh -r

# create python venv for py3-triton-linalg
ARG PY3_TRITON_LINALG=${VENV_PATH}/triton-linalg
RUN ${CONDA_PATH}/envs/triton-py3124/bin/python -m venv ${PY3_TRITON_LINALG} && \
    ${PY3_TRITON_LINALG}/bin/pip install --upgrade pip && \
    ${PY3_TRITON_LINALG}/bin/pip3 install wheel setuptools ninja wheel pybind11 torch numpy scipy pytest lit pandas matplotlib && \
    chmod 777 -R ${PY3_TRITON_LINALG}

# build triton-linalg
# RUN . ${PY3_TRITON_LINALG}/bin/activate && \
#     mkdir -p ${CODE_PATH} && cd ${CODE_PATH} && \
#     git clone --recurse-submodules -b mingfa-w/e601be5d https://github.com/mingfa-w/triton-linalg.git && \
#     cd triton-linalg && \
#     bash mingfa-w/build.sh

# == 删除过程信息 ==
RUN rm -r /root/.ssh
# 设置sshd自启动
CMD ["/usr/sbin/sshd", "-D"]
WORKDIR /host
# ENTRYPOINT ["/bin/bash"]
