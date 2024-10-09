ARG arch_version

FROM python:3.10 as base

MAINTAINER MLR <allrank@allegro.pl>

RUN mkdir /allrank
COPY requirements.txt setup.py Makefile README.md /allrank/

RUN make -C /allrank install-reqs

WORKDIR /allrank

FROM base as CPU
# https://pytorch.org/get-started/previous-versions/
#RUN python3 -m pip  install numpy==1.24.1 torchvision==0.14.1 torch==1.13.1  --extra-index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu

FROM base as GPU
# https://pytorch.org/get-started/previous-versions/
#RUN python3 -m pip  install torchvision==0.14.1 torch==1.13.1
RUN python3 -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

FROM ${arch_version} as FINAL
