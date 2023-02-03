#importing libraries
import numpy
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences

#load data
top_words = 5000
(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words=top_words)

print(X_train[1])
print(type(X_train[1]))
print(len(X_train[1]))

#truncate and/or pad input sequences
max_review_length = 600
X_train = pad_sequences(X_train, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)

print(X_train.shape)
print(X_train[1])


# Create the model
embedding_vector_length = 32

model = Sequential()
model.add(Embedding(top_words + 1, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

hist = model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(X_test, y_test))

# Final evaluation of the model
hist.history['accuracy'][-1]*1

plt.plot([i+1 for i in range(10)],hist.history['accuracy'])
plt.plot([i+1 for i in range(10)],hist.history['val_accuracy'])
plt.title('Accuracy and Val Accuracy v/s Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Accuracy','Validation Accuracy'])


plt.plot([i+1 for i in range(10)],hist.history['loss'])
plt.plot([i+1 for i in range(10)],hist.history['val_loss'])
plt.title('Loss and Val Loss v/s Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss','Validation Loss'])
