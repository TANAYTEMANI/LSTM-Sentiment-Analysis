# LSTM-Sentiment-Analysis

Sentiment Analysis is a field of Natural Language Processing (NLP) that aims to automatically determine the sentiment expressed in a text, be it positive, negative or neutral. This is typically done by using machine learning algorithms to classify the text into one of several predefined sentiment categories, based on patterns in the text data. It is widely used in various applications such as social media monitoring, brand reputation management, customer service, and opinion mining. 

LSTMs are a type of Recurrent Neural Network (RNN) that can process sequential data effectively, making them well suited for tasks such as sentiment analysis, where the order of words in a sentence can carry important sentiment information.

In a typical sentiment analysis task using LSTMs, the input is a sequence of words in a sentence, which are first transformed into numerical vectors (also known as embeddings) that can be input into the LSTM. The LSTM then processes the sequence of word vectors and outputs a single sentiment prediction for the entire sentence. The model is trained on a large dataset of labeled sentences, where the objective is to learn to predict the sentiment label correctly.

LSTMs have proven to be very effective in sentiment analysis, as they can capture complex relationships between words in a sentence and produce more accurate sentiment predictions compared to traditional machine learning methods.

## Long-Short Term Memory

### Importing Libraries
```javascript I'm A tab
import numpy
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences
```

### Load Data
```javascript I'm A tab
top_words = 5000
(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words=top_words)
```

```javascript I'm A tab
print(X_train[1])
print(type(X_train[1]))
print(len(X_train[1]))
```

### Truncate and/or pad input sequences
```javascript I'm A tab
max_review_length = 600
X_train = pad_sequences(X_train, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)

print(X_train.shape)
print(X_train[1])
```

### Create the model
```javascript I'm A tab
embedding_vector_length = 32

model = Sequential()
model.add(Embedding(top_words + 1, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

hist = model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(X_test, y_test))
```

### Final evaluation of the model
```javascript I'm A tab
hist.history['accuracy'][-1]*1
print("Accuracy: %.2f%%" % (scores[1]*100))
```

### Accuracy vs Epoch
```javascript I'm A tab
plt.plot([i+1 for i in range(10)],hist.history['accuracy'])
plt.plot([i+1 for i in range(10)],hist.history['val_accuracy'])
plt.title('Accuracy and Val Accuracy v/s Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Accuracy','Validation Accuracy'])
```
![image](https://user-images.githubusercontent.com/82306595/216568432-4b2d18ff-aab0-4374-a862-2e30816c012d.png)



### Loss vs Epoch
```javascript I'm A tab
plt.plot([i+1 for i in range(10)],hist.history['loss'])
plt.plot([i+1 for i in range(10)],hist.history['val_loss'])
plt.title('Loss and Val Loss v/s Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss','Validation Loss'])
```

![image](https://user-images.githubusercontent.com/82306595/216568506-38cd2f3b-86c5-4aba-9eb0-864f0c8787ac.png)
