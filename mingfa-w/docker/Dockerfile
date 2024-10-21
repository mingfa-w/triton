ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN echo "root:8uhb9ijn" | chpasswd
EXPOSE 22 9405 5555

RUN apt-get update && apt-get -y --no-install-recommends install \
	apt-utils \
    software-properties-common \
    build-essential \
    wget \
    dialog \
    # 调测使用
    git \
    vim \
    sudo \
    apt-transport-https \
    iputils-ping  \
    net-tools \
    openssh-client \
    openssh-server \
    gdb \
	clang lld llvm ninja-build ccache libssl-dev \
    > /dev/null && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists

# install cmake
RUN mkdir /tmp/cmake && cd /tmp/cmake && \
	wget https://cmake.org/files/v3.20/cmake-3.20.0.tar.gz && \
	tar -zxvf cmake-3.20.0.tar.gz && \
	cd cmake-3.20.0 && \
    ./configure && make -j$(nproc) && make install && \
    rm -rf /tmp/cmake
