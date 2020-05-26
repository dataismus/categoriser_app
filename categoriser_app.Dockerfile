FROM dataismus/flask:ml

COPY ./service.py /deploy_api/app.py
COPY ./code/trained.pkl /deploy_api/trained.pkl
ENTRYPOINT ["python", "/deploy_api/app.py"]