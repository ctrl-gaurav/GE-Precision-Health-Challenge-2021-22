import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json
import pickle
import os
from cxrPrediction import run_model
from chatBot import Engine

app = Flask(__name__)
app.static_folder = 'static'

nltk.download('popular')
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('Dataset/data.json').read())
words = pickle.load(open('NLP Trained Model/texts.pkl','rb'))
classes = pickle.load(open('NLP Trained Model/labels.pkl','rb'))
model = load_model('NLP Trained Model/model.h5')

bot = Engine()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():

    text = request.args.get('msg')
    return bot.chatbot_response(text, lemmatizer, intents, words, classes, model)


@app.route('/predict', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':

        # Getting the file from post request
        f = request.files['file']

        # Saving the file to ./uploads
        basepath = os.path.dirname(__file__)

        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        run_model = run_model()
        result = run_model.model_predict(img_path=file_path)

        return str(result)

    return None


if __name__ == "__main__":
    app.run()