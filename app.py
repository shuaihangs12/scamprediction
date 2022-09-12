import flask
import joblib
import pandas as pd


def init_model():
    global lgbm_model, vectorizer
    lgbm_model = joblib.load("lgb.pkl")
    vectorizer = joblib.load("vector.pkl")

init_model()

app = flask.Flask(__name__)


@app.route("/spam", methods=['POST', 'GET'])
def pred_spam():
    if flask.request.method == "POST":
        return {"result": pred_input(flask.request.form['text'])}
    if flask.request.method == "GET":
        return "please use POST"


def class_label(id):
    class_label_map = {1: "Fraud", 0: "Non Fraud"}
    return class_label_map[id]


def pred_input(text):
    tfidf_vec_k = vectorizer.transform([text])
    # Get prediction
    tfidf_data = pd.DataFrame(tfidf_vec_k.toarray())
    output = class_label(lgbm_model.predict(tfidf_data)[0])
    print(output)
    return output
