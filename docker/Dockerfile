# syntax=docker/dockerfile:experimental
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
	DEBIAN_FRONTEND=noninteractive \
	PYTHONWARNINGS="ignore" \
	TZ=Asia/Taipei

# Install and update tools to minimize security vulnerabilities
RUN apt update -y && apt install -y software-properties-common wget apt-utils patchelf git git-lfs libprotobuf-dev protobuf-compiler cmake \
	git bash curl libturbojpeg exiftool ffmpeg poppler-utils libheif-dev \
	libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
	libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
	libharfbuzz-dev libfribidi-dev libxcb1-dev
RUN unattended-upgrade
RUN apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# install miniconda (comes with python 3.10 default)
ARG MINICONDA_PREFIX=/home/user/miniconda3

ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
RUN curl -fSsL --insecure ${CONDA_URL} -o install-conda.sh && \
	/bin/bash ./install-conda.sh -b -p $MINICONDA_PREFIX && \
	$MINICONDA_PREFIX/bin/conda clean -ya && \
	$MINICONDA_PREFIX/bin/conda install -y python=3.10

ENV PATH=$MINICONDA_PREFIX/bin:${PATH}

ARG PYTHON_EXE=$MINICONDA_PREFIX/bin/python
RUN pip install -U pip wheel
RUN pip install torch torchvision
RUN MAX_JOBS=4 MMCV_WITH_OPS=1 pip install git+https://github.com/open-mmlab/mmcv.git@v2.1.0

COPY requirements_linux.txt .
RUN pip install -r requirements_linux.txt && rm requirements_linux.txt
RUN python -c "import mmcv, mmdet, capybara, chameleon"
RUN pip install ipython

ARG work_folder
WORKDIR $work_folder