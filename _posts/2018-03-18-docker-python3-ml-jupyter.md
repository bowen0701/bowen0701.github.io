---
layout: post
comments: true
title: "Docker for Machine Learning with Python3"
excerpt: "Docker for machine learning with Python3 and Jupyter Notebook."
date: 2018-03-18
mathjax: true
---

## Introduction

[Wikipedia:](https://en.wikipedia.org/wiki/Docker_(software)) "Docker is a software technology providing containers, promoted by the company Docker, Inc. Docker provides an additional layer of abstraction and automation of operating-system-level virtualization on Windows and Linux. Docker uses the resource isolation features of the Linux kernel such as cgroups and kernel namespaces, and a union-capable file system such as OverlayFS and others to allow independent “containers” to run within a single Linux instance, avoiding the overhead of starting and maintaining virtual machines (VMs)."

**Why Docker for Machine Learning?** We data scientists / machine learning engineers would like to solve data-driven problems and build machine learning-based products. Using docker to create (1) reproducibility, (2) compute environment portability, and (3) engineering applications, for example RESTful API for serving machine learning predictions, is a popular and wise choice. 

**Why Python?** Python is one of best languages for building data science applications, it is general purpose and engineering-oriented, with fairly good performance, making data preprocessing, machine learning and data visualizations easy, efficient and flexible. Nowadays, lots of top Internet services are built upon Python, including YouTube, Instegram, Quora, among many others.

**Why Python3?** Honestly, I use Python2 previously, and recently I knew that by the end of 2019, the scientific stack will stop supporting Python2. As for the most important scientific computing package in Python, **Numpy,** all of new feature releases will only support Python3 after 2018. Hence I decided to switch from Python2 to Python3, via docker!

## Docker Terminology

- **Image:** is essentially blueprint of the containers for what we would like to build.
- **Container:** is an instantiation of an image (For us Pythonistas, we can think image is a class, and container is the class's initialized object).
- **Dockerfile:** is a setup file to create/modify Docker images.
- **Docker-Compose:** is a tool for us to easily define and launch Docker applications.

## Docker Setup

For access the scripts please refer to my corresponding GitHub repo: [https://github.com/bowen0701/docker-python3-ml-jupyter](https://github.com/bowen0701/docker-python3-ml-jupyter).

### Install Docker

We can download and install docker on [docker website](https://www.docker.com/community-edition#/download), following the instructions we can finish the procedure easily. I choose Docker CE for Mac version. After installing we now have docker and docker-compose.

### Create Dockerfile

Many people leverage dockers contributed from many others, this is convenient and efficient for us to ramp up fast. Nevertheless, I prefer making my own docker (almost) from scratch, from this experience I can learn a lot and acquire fundamental knowledge of docker.

The first step is to create our Dockerfile; for details please refer to [Dockerfile](ttps://github.com/bowen0701/docker-python3-ml-jupyter/blob/master/Dockerfile). 

```
# Docker settings: Ubuntu, Python3, pip, general machine learning frameworks, Jupyter Notebook.

FROM ubuntu:16.04

LABEL maintainer="Bowen Li <bowen0701@gmail.com>"

# Pick up some Python3 dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        # python \
        # python-dev \
        # python-pip \
        # python-setuptools \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python3 general packages.
RUN pip3 install --upgrade pip \
        numpy \
        scipy \
        pandas \
        sklearn \
        ipykernel \
        jupyter \
        matplotlib \
        seaborn \
        Cython \
        Pillow \
        requests \
        awscli \
        && \
    python3 -m ipykernel.kernelspec

# Install machine learning packages.
RUN pip3 --no-cache-dir install --upgrade \
        tensorflow \
        tensorflow-tensorboard
        # http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl \
        # torchvision \
        # keras \
        # xgboost \
        # pymc3 \
        # pystan \
        # gensim \
        # nltk \
        # opencv-python

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Copy sample notebooks.
COPY notebooks /notebooks

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_cmd.sh /

# Jupyter Notebook
EXPOSE 8888
# TensorBoard
EXPOSE 6006

WORKDIR /notebooks

RUN chmod +x /run_cmd.sh

CMD ["/run_cmd.sh"]
```

The Dockerfile is based on basic docker image for OS only, ubuntu:16.04. Then we install Python3's general packages, including 

- `Numpy`
- `Scipy`
- `Pandas`
- `Scikit-learn`
- `Jupyter`
- `Matplotlib`
- `Seaborn`
- `Cython`, etc, 

and machine learning frameworks, including 

- `PyTorch`
- `TensorFlow`
- `Keras`
- `XGBoost`
- `PyMC`
- `PyStan`
- `Gensim`
- `NLTK`, and 
- `OpenCV`.

Finally, the remaining is to set up Jupyter Notebook. Note that we expose docker's port 8888 to Jupyter Notebook.

Note that of courese we can add/delete any general / machine learning packages in Dockerfile, as our needs.

### Create docker-compose.yml

To apply scripts as environment, we could create docker-compose.yml for us to easily launch docker; for details please refer to [docker-compose.yml](ttps://github.com/bowen0701/docker-python3-ml-jupyter/blob/master/docker-compose.yml). 

```
version: '3'
services:
  web:
    build: .
    image: docker-ml:latest
    ports:
     - "8888:8888"
     - "6006:6006"
    volumes:
     - ./notebooks:/notebooks
```

In this file we assign what docker image (docker-ml:latest) we would like to use, what ports (8888:8888 and 6006:6006) to connect Jupyter Notebook and TensorBoard respectively, and what volums (./notebooks:/notebooks) to share data between our docker container and the host computer. Note that we use docker-compose.yml's version 3 format.

### Create jupyter_notebook_config.py

In this file, we set up Jupyter Notebook's IP, port and password, if set in environment; for details please refer to [jupyter_notebook_config.py](ttps://github.com/bowen0701/docker-python3-ml-jupyter/blob/master/jupyter_notebook_config.py). 

```
import os
from IPython.lib import passwd

# Set Jupyter Notebook's IP & port.
c.NotebookApp.ip = '*'
c.NotebookApp.port = int(os.getenv('PORT', 8888))
c.NotebookApp.open_browser = False

# Set a password if PASSWORD is set in the environment
if 'PASSWORD' in os.environ:
    c.NotebookApp.password = passwd(os.environ['PASSWORD'])
    del os.environ['PASSWORD']
```

This jupyter_notebook_config.py will be copied to replace the original one in /root/.jupyter/. Note that basically we do not have to edit this file anymore.

### Create run_cmd.sh

Finally, in this file we collect bash scripts to launch Jupyter Notebook; for details please refer to [run_cmd.sh](ttps://github.com/bowen0701/docker-python3-ml-jupyter/blob/master/run_cmd.sh). 

```
#! /bin/bash

jupyter notebook --no-browser --allow-root --port 8888 --ip 0.0.0.0 --NotebookApp.token='' &
tensorboard --port 6006 --ip 0.0.0.0 --logdir /logs
```

Note that

- `--allow-root`: is needed, since we use root to launch docker.
- `--port 8888`: must be the same as docker's port for Jupyter Notebook.
- `--ip 0.0.0.0`: reassign Jupyter Notebook server's IP.
- `--NotebookApp.token=''`: optional, for simplicity we here disable the toker authentification. Note that this is not recommended for security reasons. Nevertheless, since I choose this to test launch on my local laptop I think it is ok for now.

To lanch TensorBoard, the pipielined bash scripts follow similarly.

## Launch Docker

### Build Docker Image

First build docker image with name `docker-ml:latest`, by docker CLI or docker-compose with Dockerfile.

```
# Build docker image by Docker CLI.
sudo docker build -t docker-ml:latest .

# Alternative, (re-)build by docker-compose, if needed.
docker-compose build
```

Now we can check docker images.

```
sudo docker images
```

### Create Docker Container

First based on newly built docker image, `docker-ml:latest`, to create docker container.

```
# Run docker image to create a container, by docker CLI.
sudo docker run -it -p 8888:8888 -p 6006:6006 docker-ml:latest
# Run in background mode.
sudo docker run -dt -p 8888:8888 -p 6006:6006 docker-ml:latest

# Alternatively, run using docker-compose with docker-compose.yaml.
docker-compose up
# Run docker compose in background.
docker-compose up -d
```

Then we can check docker container status:

```
docker ps -a
```

**Remarks:**

- If cannot access jupyter notebook server on browser, maybe port 8888 is ocupied by some unknown process, run this:

```
lsof -ti:8888 | xargs kill -9
```

- If your docker-compose is abnormally slow, add one of these in /etc/hosts by `sudo vim /etc/hosts`.

```
127.0.0.1 localunixsocket
127.0.0.1 localunixsocket.local
```

### Stop/Restart Docker Container

```
# Stop/restart docker container
sudo docker stop <docker_container_id>
sudo docker restart <docker_container_id>

# Stop/restart by docker compose.
docker-compose stop
docker-compose restart
docker-compose down
```

### Access Jupyter Notebook 

Finally, we can access Jupyter Notebook server from the browser at this URL:

```
http://0.0.0.0:8888
```

### Access TensorBoard

```
http://0.0.0.0:6006
```

Now enjoy your docker for machine learning with Python3 and Jupyter Notebook. :-)

### Go into Bash Mode

```
# Go into /bin/bash mode.
docker exec -it <docker_container_id> /bin/bash

# Exit bash mode.
exit
```

### Remove Docker Container / Image

```
# Remove docker container.
sudo docker rm <docker_container_id>

# Remove docker images.
sudo docker rmi <docker_image_id>
```

## Share Our Image to DockerHub

We can rebuild our local images following the naming rule, `<hub-user>/<repo-name>[:<tag>]`, or re-tagging an existing local image.

```
# Rebuild docker image.
docker build -t bowen0701/docker-python3-ml-jupyter

# Retag docker image.
docker tag docker-ml bowen0701/docker-python3-ml-jupyter
```

The final step is to push renamed docker image to DockerHub.

```
docker push bowen0701/docker-python3-ml-jupyter
```

## References

- https://store.docker.com/community/images/tensorflow/tensorflow
- https://store.docker.com/community/images/dash00/tensorflow-python3-jupyter
- https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5
