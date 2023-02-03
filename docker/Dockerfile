FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV TZ Asia/Seoul
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get -y install python3 \
    python3-pip \
    python3-dev \
    mysql-client libmysqlclient-dev python3-mysqldb \
    git ssh vim \
    libsm6 libxext6 libxrender-dev

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN echo "root:password" | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

RUN pip3 install --upgrade pip
RUN pip3 install setuptools

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

ENV DJANGO_SUPERUSER_USERNAME djangoadmin
ENV DJANGO_SUPERUSER_EMAIL none@none.com
ENV DJANGO_SUPERUSER_PASSWORD djangopwdpwd
ENV C_FORCE_ROOT = "true"

RUN export DJANGO_SUPERUSER_USERNAME=djangoadmin
RUN export DJANGO_SUPERUSER_EMAIL=none@none.com
RUN export DJANGO_SUPERUSER_PASSWORD=djangopwdpwd
RUN export C_FORCE_ROOT="true"
RUN export SERVICE_NAME_DATABASE=database
RUN export MYSQL_ROOT_PASSWORD=dbpwdpwd
RUN export MYSQL_DATABASE=dbname
RUN export MYSQL_USER=dbadmin
RUN export MYSQL_PASSWORD=dbpwdpwd

WORKDIR /workspace
ADD . .
RUN pip install -r requirements.txt
RUN mkdir /workspace/static

RUN service ssh restart
