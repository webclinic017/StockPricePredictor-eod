FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

EXPOSE 80

ENV NAME World

CMD ["python","01_main.ipynb"]