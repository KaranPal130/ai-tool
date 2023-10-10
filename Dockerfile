FROM nvidia/cuda:11.8.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git wget 
RUN apt-get install -y python3 python3-dev python3-pip openssh-server

RUN pip3 install --upgrade pip

WORKDIR     /home

COPY        ./requirements.txt .

COPY        * .

RUN         pip3 install -r requirements.txt

USER        root

EXPOSE      8000

CMD         ["uvicorn", "main:app", "--port", "8000", "--host", "0.0.0.0"]