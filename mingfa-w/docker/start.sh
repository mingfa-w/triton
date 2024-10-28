#!/bin/bash -xe
script_dir=$(cd $(dirname $0); pwd)
cur_dir=`pwd`
cd ${script_dir}
. ./common.sh
CONTAINER_NAME=$USER.$tag
STOP=0
PORT=9006

# 处理参数
while getopts "su:p:" opt
do
    case $opt in
        s)
            echo "选项 -s(stop) 被设置"
            STOP=1
            ;;
        u)
            echo "选项 -u(user) 的值是 $OPTARG"
            CONTAINER_NAME=$OPTARG.$tag
            ;;
        p)
            echo "选项 -p(port) 的值是 $OPTARG"
            PORT=$OPTARG
            ;;
        \?)
            echo "无效选项: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# docker_in_docker=" --net=host --privileged -v /var/run/docker.sock:/var/run/docker.sock -v $(which docker):/bin/docker "
docker_in_docker=" -p $PORT:22 --privileged -v /var/run/docker.sock:/var/run/docker.sock -v $(which docker):/bin/docker "
# echo $docker_in_docker ====; exit 0
# docker_run_flag=" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --cap-add=SYS_PTRACE --security-opt seccomp=unconfined "
docker_run_flag=" --gpus all "

MOUNT_DIR=$HOME
MOUNT_DIR_A800="  -v /data00:/data00 -v /data01:/data01 -v /data02:/data02 -v /data03:/data03 \
    -v /data04:/data04 -v /data05:/data05 -v /data06:/data06 -v /data07:/data07 "

GROUP=`id -g -n`
GROUPID=`id -g`
OLD_ID=`docker ps -aq -f name=$CONTAINER_NAME -f status=running`
echo ==== container name: $CONTAINER_NAME

# if [[ "$1 " == "show" ]]; then
#     exit
# fi

# if [[ "$1" == "root" ]]; then
#     echo === root ===
#     docker exec -it --user root $CONTAINER_NAME bash
#     exit
# fi
if [[ $STOP == 1 ]]; then
    echo === stop $CONTAINER_NAME
    docker stop $CONTAINER_NAME
    exit 0
fi

if [ -z "$OLD_ID" ]; then
    ID=`docker run $docker_in_docker $docker_run_flag -t -d --name $CONTAINER_NAME $MOUNT_DIR_A800 -v $MOUNT_DIR:/host --tmpfs /tmp:exec --rm $image `
    docker exec --user root $ID groupadd -f -g $GROUPID $GROUP
    docker exec --user root $ID adduser --shell /bin/bash --uid $UID --gecos '' --ingroup $GROUP --disabled-password --home /home/$USER --force-badname $USER
    #docker exec --user root $ID bash -c " echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER && chmod 0440 /etc/sudoers.d/$USER"
    userPasswd="8uhb9ijn"
    echo -e "$userPasswd\n$userPasswd" | docker exec --user root -i $ID passwd $USER
    docker exec --user root $ID usermod -aG sudo $USER
    docker exec --user root $ID bash -c " echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers"
    docker exec --user root $ID bash -c ' mkdir ~/.ssh; mkdir /tmp/ccache; ln -s /tmp/ccache ~/.ccache'

    docker exec --user $USER $ID bash -c "mkdir -p /home/$USER/.ssh"
    if [ -f ~/.ssh/authorized_keys ]; then
        docker cp ~/.ssh/authorized_keys $ID:/home/$USER/.ssh/
    fi

    if [ -f ~/.ssh/config ]; then
        docker cp ~/.ssh/config $ID:/home/$USER/.ssh/
    fi

    if [ -f ~/.ssh/id_rsa.pub ]; then
        docker cp ~/.ssh/id_rsa.pub $ID:/home/$USER/.ssh/
    fi

    if [ -f ~/.ssh/id_rsa ]; then
        docker cp ~/.ssh/id_rsa $ID:/home/$USER/.ssh/
    fi

    if [ -f ~/.ssh/known_hosts ]; then
        docker cp ~/.ssh/known_hosts $ID:/home/$USER/.ssh/
    fi
    

    GIT_USER=`git config --get user.name`
    GIT_EMAIL=`git config --get user.email`
    docker exec --user $USER $ID bash -c "git config --global user.name \"$GIT_NAME\""
    docker exec --user $USER $ID bash -c "git config --global user.email \"$GIT_EMAIL\""
    #docker cp ./gitconfig $ID:/home/$USER/.ssh

    docker exec --user root $ID bash -c "cat /proc/1/environ > /tmp/proc_env"
    docker exec --user root $ID bash -c "chmod 777 /tmp/proc_env"
    # docker exec --user $USER $ID bash -c "cat /get_proc_env >> ~/.bashrc"
    # docker exec --user root $ID bash -c "rm /get_proc_env"

    # golang config
    if [ -f ~/.gitconfig ]; then
        docker cp $HOME/.gitconfig $ID:/home/$USER/
    fi

    #docker cp $HOME/.ssh/config $ID:/home/$USER/.ssh/config
    # docker exec --user $USER $ID bash -c "go env -w GOPROXY=\"https://go-mod-proxy.byted.org,https://goproxy.cn,https://proxy.golang.org,direct\""
    # docker exec --user $USER $ID bash -c "go env -w GOPRIVATE=\"*.byted.org,*.everphoto.cn,git.smartisan.com\""
    # docker exec --user $USER $ID bash -c "go env -w GOSUMDB=\"sum.golang.google.cn\""
    # docker exec --user root $ID bash -c "go env -w GOPATH=\"\$HOME/go\""
    # docker exec --user $USER $ID bash -c "go env -w GOMODCACHE=\"\$HOME/go\""
    
fi
cd ${cur_dir}
docker exec -it --user $USER $CONTAINER_NAME bash
