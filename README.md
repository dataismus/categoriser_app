docker run -d --rm -p 5000:5000 --name flask_categoriser dataismus/categoriser_app

curl -XPOST 'http://0.0.0.0:5000/classify' -H 'Content-Type: application/json' -d '{ "title": "my product title containing aquarelle" }'
