ARG IMAGE_NAME
FROM nvidia/cuda:10.0-runtime-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV CUDNN_VERSION 7.6.5.32
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
&& \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y python3-pip python3.6-dev \
 && cd /usr/local/bin \
 && ln -s /usr/bin/python3.6 python \
 && pip3 install --upgrade pip \
 && apt-get install -y build-essential cmake \
 && apt-get install -y libgtk-3-dev \
 && apt-get install -y libboost-all-dev

WORKDIR /usr/scr/app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD python website/manage.py runserver 0.0.0.0:8000

COPY . .
