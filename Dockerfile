FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
LABEL maintainer="ssamine"
LABEL repository="medi_coqa"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   wget \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch==1.4.0

WORKDIR /workspace
COPY . transformers-coqa/
RUN cd transformers-coqa/ && \
    pip3 install -r requirements.txt && \
    python3 -m spacy download en && \
    wget https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json -O data/coqa-train-v1.0.json && \
    wget https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json -O data/coqa-dev-v1.0.json

CMD ["/bin/bash"]
