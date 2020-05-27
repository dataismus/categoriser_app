import pickle
import json
from flask import Flask, request
import numpy as np
service = Flask(__name__)

##################

def unpickle():
    global trained
    global classes
    with open('/deploy_api/trained.pkl', 'rb') as f: trained = pickle.load(f)
    classes=trained.classes_

##################

@service.route('/classify', methods=['POST'])
def get_prediction():
    data = request.get_json()
    probas=trained.predict_proba([data["title"]])[0]
    top5=np.argsort(probas)[-5:]

    #requested JSON elements:
    title=data["title"]
    top_5_results=[{"product_type": x, "score": "{:.4f}".format(y)} for (x,y) in zip(reversed(classes[top5]),reversed(probas[top5]))]
    product_type=classes[top5][-1]
    return json.dumps({"title": title, "top_5_results": top_5_results, "product_type": product_type}, indent=3)

##################

if __name__ == '__main__':
    unpickle()
    service.run(host='0.0.0.0', port=5000)