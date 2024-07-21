# docker tag local-image:tagname new-repo:tagname
# docker push new-repo:tagname

# Using Ubuntu 22.04
FROM ubuntu:22.04

# Env setting non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# updata and install toolkits
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && apt-get clean 

RUN pip install yq

# setting default python version 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# install requirements python toolkits
RUN python3 -m pip install -r requirements.txt

# install pytorch
RUN pip3 install --no-cache-dir torch torchvision torchaudio

# 設置工作目錄
WORKDIR /workspace

# 拷貝當前目錄的內容到工作目錄
COPY . /workspace

RUN chmod +x ./run_script.sh

# 設置容器啟動時的默認命令（可根據實際需要調整）
CMD ["./run_script.sh"]