FROM nvidia/cudagl:10.2-runtime-ubuntu18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN conda install -c conda-forge igl=2.2.0
COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt
RUN apt-get update && apt-get install -y git build-essential
RUN git clone https://github.com/tristandeleu/pytorch-meta.git && cd pytorch-meta && python setup.py install
RUN pip install setuptools==59.5.0
RUN apt-get update && apt-get install -y freeglut3-dev
