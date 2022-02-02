import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.models import load_model
import random
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from cxrPrediction import run_model




class Engine():
    def __init__(self) -> None:
        pass


    @staticmethod
    def clean_up_sentence(sentence, lemmatizer):

        # tokenizing the pattern - splitting words into array
        sentence_words = nltk.word_tokenize(sentence)

        # stemming each word - creating short form for word
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

        return sentence_words


    # function to return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    @staticmethod
    def bow(sentence, words, lemmatizer, show_details=True):

        # ing the pattern
        sentence_words = Engine.clean_up_sentence(sentence, lemmatizer)

        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    # assigning 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)

        return(np.array(bag))

    
    @staticmethod
    def predict_class(sentence, model, words, classes, lemmatizer):

        # filtering out predictions below a threshold
        p = Engine.bow(sentence, words, lemmatizer, show_details=False)

        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

        # sorting by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)

        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list


    @staticmethod
    def response(ints, intents_json):

        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']

        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break

        return result

    @staticmethod
    def chatbot_response(msg, lemmatizer, intents, words, classes, model):

        ints = Engine.predict_class(msg, model, words, classes, lemmatizer)
        res = Engine.response(ints, intents)
        
        return res