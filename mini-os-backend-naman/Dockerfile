FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git curl && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install \
    numpy pandas matplotlib scikit-learn seaborn \
    jupyter notebook tensorflow torch torchvision transformers opencv-python

CMD [ "bash" ]
