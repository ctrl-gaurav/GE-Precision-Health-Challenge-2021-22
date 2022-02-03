import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences




model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(padded_sequences, np.array(training_labels), epochs=100)


# saving model
model.save("chat_model.h5")

