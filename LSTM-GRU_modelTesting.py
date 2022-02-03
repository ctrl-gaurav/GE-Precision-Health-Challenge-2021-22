import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import pickle




class faces():
    def __init__(self) -> None:
        pass

    def model_predict(self, img_path):



with open("intents.json") as file:
    data = json.load(file)

model = keras.models.load_model('chat_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)


max_len = 100

inp = input()

result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
truncating='post', maxlen=max_len))

# probability
x=np.argmax(result)
result[0][x]

tag = lbl_encoder.inverse_transform([np.argmax(result)])

for i in data['intents']:
    if i['tag'] == tag:
        print(tag)