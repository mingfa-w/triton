# This is a script to help install LLVM 11.0.1
current_dir=$(cd $(dirname $0); pwd)
export rebuild=$1 # 0-重新编译llvm-project
TARGET_ARCH=`uname -m`
TRITON_MAJ_VER=d39ee1f9

# llvm-project version
LLVM_MAJ_VER=triton-4bb928ec-llvm-c08c6a71
llvm_install_dir=${HOME}/.local/llvm-$LLVM_MAJ_VER-$TARGET_ARCH

# triton path
triton_src_dir=${current_dir}/..
triton_build_dir=${current_dir}/build
triton_install_dir=${HOME}/.local/triton-$TRITON_MAJ_VER-$TARGET_ARCH
triton_dir=${triton_install_dir}/lib/cmake/triton

function build_triton(){
    if [ ! -d ${triton_src_dir/python/build} ]; then
        rm -rf ${triton_src_dir/python/build}
    fi
    cd ${triton_src_dir}
    # source $HOME/miniconda3/etc/profile.d/conda.sh
    # conda activate triton-py3124
    source $HOME/apps/venv/py3112/bin/activate
    pip install ninja cmake wheel pybind11 # build-time dependencies
    
    LLVM_INCLUDE_DIRS=${llvm_install_dir}/include \
    LLVM_LIBRARY_DIR=${llvm_install_dir}/lib \
    LLVM_SYSPATH=${llvm_install_dir} \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_WITH_CCACHE=true \
    pip install -e python --no-build-isolation
    cd ${current_dir}
}

build_triton

echo "triton build done"
