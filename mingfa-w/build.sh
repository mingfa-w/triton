current_dir=$(cd $(dirname $0); pwd)
export rebuild=0 # 1-重新编译
TARGET_ARCH=`uname -m`
TRITON_MAJ_VER=d39ee1f9

# llvm-project version
LLVM_MAJ_VER=triton-d39ee1f9-llvm-61f8a7f6
# llvm_install_dir=${HOME}/.local/llvm-$LLVM_MAJ_VER-$TARGET_ARCH
llvm_install_dir=/opt/llvm_${LLVM_MAJ_VER}
echo llvm_install_dir=$llvm_install_dir
# python venv
py_venv_path=/opt/apps/py3-triton/bin/activate

# triton path
triton_src_dir=${current_dir}/..
triton_build_dir=${current_dir}/build
triton_install_dir=${HOME}/.local/triton-$TRITON_MAJ_VER-$TARGET_ARCH
triton_dir=${triton_install_dir}/lib/cmake/triton

# 处理参数
while getopts "rp:" opt
do
    case $opt in
        r)
            echo "选项 -r 被设置"
            rebuild=1
            ;;
        l)
            echo "选项 -l(llvm_install_dir) 的值是 $OPTARG"
            llvm_install_dir=$OPTARG
            ;;
        \?)
            echo "无效选项: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

function build_triton(){
    if [ ! -d ${triton_src_dir/python/build} ]; then
        rm -rf ${triton_src_dir/python/build}
    fi
    cd ${triton_src_dir}
    # source $HOME/apps/venv/python3112-triton/bin/activate
    source $py_venv_path
    # pip install ninja cmake wheel pybind11 # build-time dependencies
    # 设置Degun模式
    export DEBUG=1
    LLVM_INCLUDE_DIRS=${llvm_install_dir}/include \
    LLVM_LIBRARY_DIR=${llvm_install_dir}/lib \
    LLVM_SYSPATH=${llvm_install_dir} \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_WITH_CCACHE=true \
    TORCH_USE_CUDA_DSA=true \
    pip install -e python --no-build-isolation
    cd ${current_dir}
}

build_triton

echo "triton build done"
