import numpy as np
from tensorflow.keras.optimizers import SGD





class ModelTraining():
    def __init__(self) -> None:
        pass

    def train(self, model, train_x, train_y):
        # Compiling model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #fitting and saving the model 
        hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        model.save('NLP Trained Model/model.h5', hist)

        print("Model Trained and Saved!")
        return model