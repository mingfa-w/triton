- 安装conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh


- 编译llvm-projects


- 安装pytorch
<!-- conda install pytorch torchvision torchaudio cudatoolkit=11 -c pytorch -->
<!-- conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge -->
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
import torch
torch.__version__
torch.cuda.is_available()