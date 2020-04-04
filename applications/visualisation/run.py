import random
import requests

from flask import Flask, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder="./dist/static", template_folder="./dist")
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/api/random")
def random_number():
    response = {"randomNumber": random.randint(1, 100)}
    return jsonify(response)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    if app.debug:
        return requests.get("http://localhost:8080/{}".format(path)).text
    return render_template("index.html")
