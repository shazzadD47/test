FROM python:3.11-slim-bookworm

WORKDIR /code
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    default-jdk \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    pandoc && \
    # Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

RUN chmod +x ./scripts/run_prod.sh

EXPOSE 8000

CMD ["./scripts/run_prod.sh"]
