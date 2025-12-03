#!/usr/bin/env bash

apt-get install -y python python-pip python-dev

pip install subprocess32 gradescope-utils

apt-get update

apt-get install -y software-properties-common python-software-properties

add-apt-repository ppa:deadsnakes/ppa

apt-get update

apt update

apt install -y python3.6

apt install -y python3.6-dev

apt install -y python3.6-venv

apt install -y python3-numpy

pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl

pip3 install numpy scipy