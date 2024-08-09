# docker tag local-image:tagname new-repo:tagname
# docker push new-repo:tagname

# Using Ubuntu 22.04
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
# FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Env setting non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Env setting 
RUN mkdir -p /mount/nfs
ENV MOUNT_PATH=/mount/nfs

# updata and install toolkits
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    jq \
    && apt-get clean 

RUN pip install yq

# setting default python version 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1


WORKDIR /workspace

COPY . /workspace

RUN chmod +x ./run_script-tpe1.sh
RUN chmod +x ./run_script.sh

# Version
ENV _VERSION=1.06

CMD ["/bin/bash"]
#CMD ["./run_script.sh"]