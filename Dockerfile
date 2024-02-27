FROM python:3.12-slim-bookworm

WORKDIR /app

COPY . /app

RUN apt update && apt install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/* && apt-get clean  \
    && pip install --no-cache-dir -r requirements.txt

CMD [ "python", "main.py" ]