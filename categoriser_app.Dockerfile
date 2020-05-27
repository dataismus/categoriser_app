# docker run -d --rm -p 5000:5000 --name flask_ricardo  dataismus/categoriser_app

FROM dataismus/flask:ml

COPY ./service.py /deploy_api/app.py
COPY ./code/trained.pkl /deploy_api/trained.pkl

ENTRYPOINT ["python", "/deploy_api/app.py"]

