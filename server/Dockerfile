FROM python:3.7.3-slim
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
WORKDIR /server
COPY requirements.api.txt .
RUN pip3 install -r requirements.api.txt
COPY . ./
CMD python3 server.py