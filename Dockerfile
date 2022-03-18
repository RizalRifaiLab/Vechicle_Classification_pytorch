
FROM python:3.7.12-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
     libglib2.0-0 libsm6 libxrender1 libxext6 && \
     rm -rdf /var/lib/apt/lists/*
RUN pip install --upgrade pip 

WORKDIR /app
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY vehicleClassification.py .
COPY cardataset_model_7.pt .

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)  

ENTRYPOINT ["python3","vehicleClassification.py"]