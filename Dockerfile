FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel

WORKDIR /home/ryan/repos-developing/Pointnet2_PyTorch
COPY requirements.txt /home/ryan/repos-developing/Pointnet2_PyTorch/

RUN pip install -r requirements.txt
RUN pip install matplotlib jupyter

RUN apt-get update && apt-get install -y \
    wget \ 
    unzip