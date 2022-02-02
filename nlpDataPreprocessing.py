import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import random

nltk.download('punkt')
nltk.download('wordnet')

class DataProcessing():
    def __init__(self) -> None:
        pass

    def process(self, data_path, texts_path, labels_path):

        words=[]
        classes = []
        documents = []
        ignore_words = ['?', '!']

        #load JSON file
        data_file = open(data_path).read()
        intents = json.loads(data_file)


        for intent in intents['intents']:
            for pattern in intent['patterns']:

                #tokenizing each word
                w = nltk.word_tokenize(pattern)
                words.extend(w)
                documents.append((w, intent['tag']))

                # adding classes list
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])



        # lemmaztize and lower each word and remove duplicates
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
        words = sorted(list(set(words)))

        # sorting classes
        classes = sorted(list(set(classes)))

        #printing created data

        # documents = combination between patterns and intents
        print (len(documents), " documents")
        # classes = intents
        print (len(classes), " classes", classes)
        # words = all words, vocabulary
        print (len(words), " unique lemmatized words", words)

        #saving data
        pickle.dump(words,open(texts_path,'wb'))
        pickle.dump(classes,open(labels_path,'wb'))


        # creating our training data
        training = []
        # creating an empty array for our output
        output_empty = [0] * len(classes)

        # training set, bag of words for each sentence

        for doc in documents:

            # initializing our bag of words
            bag = []

            # list of tokenized words for the pattern
            pattern_words = doc[0]

            # lemmatizing each word - create base word, in attempt to represent related words
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

            # creating our bag of words array with 1, if word match found in current pattern
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)
            
            # output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            
            training.append([bag, output_row])


        # shuffling our features and turning into a numpy array

        random.shuffle(training)
        training = np.array(training)

        # creating train and test lists. X - patterns, Y - intents
        train_x = list(training[:,0])
        train_y = list(training[:,1])
        print("Training data created")
        
        return train_x, train_y