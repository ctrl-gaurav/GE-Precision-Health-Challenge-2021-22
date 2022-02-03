from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, GRU



class nlpModel():
    def __init__(self) -> None:
        pass

    def model(self, img_path):

        model = Sequential()

        model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))

        model.add(GRU(units = 256, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(GRU(units = 128, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(GRU(units = 128, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 64))
        model.add(Dropout(0.2))

        model.add(Dense(num_classes, activation='softmax'))

        return model