# 1. 创建新环境
conda create -n multiagent python=3.10 -y
conda activate multiagent

# 2. 设置清华镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes

# 3. 安装依赖（推荐部分用 pip）
pip install mujoco==3.2.2 dm_control numpy scipy matplotlib tqdm
