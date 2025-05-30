FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

USER root

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    wget \
    git \
    && apt-get clean

WORKDIR /workspace/experiment
COPY requirements.txt /workspace/experiment

# Install an SSH server
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir /var/run/sshd

# Allow root login & password auth (use your own secure password!)
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "root:root" | chpasswd

EXPOSE 22

RUN pip3 install -r requirements.txt
RUN pip3 install flash-attn --no-build-isolation
RUN pip install torchvision

COPY . /workspace/experiment
ENV PYTHONPATH=/workspace/experiment

WORKDIR /workspace/experiment
ENTRYPOINT ["bash", "docker-run.sh"]
