from nlpDataPreprocessing import DataProcessing
from nlpNNarchitecture import Architecture
from nlpTrain import ModelTraining
import os


# Setting Directories
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'NLP Trained Model')
if not os.path.exists(final_directory):
   os.makedirs(final_directory)


# Setting Paths
data_path = 'Dataset/data.json'
texts_path = 'NLP Trained Model/texts.pkl'
labels_path = 'NLP Trained Model/labels.pkl'

# Pre-Processing Data
preProcessData = DataProcessing()
train_x, train_y = preProcessData.process(data_path, texts_path, labels_path)

# Intializing Model Architecture
nnArchitecture = Architecture()
model = nnArchitecture.model(train_x, train_y)

# Compiling and Training Deep Neural Network
trainModel = ModelTraining()
trainModel.train(model, train_x, train_y)
