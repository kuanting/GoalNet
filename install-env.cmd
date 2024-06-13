conda create -y --name goalnet python=3.11
conda activate goalnet

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install pandas opencv-python tqdm matplotlib pyyaml scipy pretrainedmodels efficientnet_pytorch
#pip install fvcore

# Install jupyter in Anaconda Navigator by yourself
