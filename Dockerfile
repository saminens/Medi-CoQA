FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
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
    mkl

WORKDIR /workspace
COPY . Medi-CoQA/
RUN cd Medi-CoQA/ && \
    pip3 install -r requirements.txt && \
    python3 -m spacy download en && \
    wget https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json -O data/coqa-train-v1.0.json && \
    wget https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json -O data/coqa-dev-v1.0.json

# uninstall Apex if present, twice to make absolutely sure :)
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

CMD ["/bin/bash"]
