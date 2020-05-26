# docker run -d --rm -p 5050:5050 dataismus/categoriser_app
# http://0.0.0.0:5050/

FROM dataismus/flask:ml

COPY ./service.py /deploy_api/app.py
COPY ./code/trained.pkl /deploy_api/trained.pkl

EXPOSE 5050

ENTRYPOINT ["python", "/deploy_api/app.py"]

