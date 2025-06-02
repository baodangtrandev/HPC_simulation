FROM python:3.7-slim

COPY requirements.txt /data/requirements.txt

RUN ["pip", "install", "-r", "/data/requirements.txt"]

CMD [ "pybatsim","/home/code/simulation/schedulers/easy.py","-t","99999999" ]