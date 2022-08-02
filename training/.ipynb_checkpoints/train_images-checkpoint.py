import os
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import pickle5 as pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.layers import Dense,Dropout,MaxPooling2D, Flatten, Conv2D


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    
    with open("img_train.pkl", "rb") as fh:
        data = pickle.load(fh)
    X_train_ = pd.DataFrame(data)["img_array"] 
    
    with open("img_test.pkl", "rb") as fh:
        data = pickle.load(fh)
    X_test_ = pd.DataFrame(data)["img_array"]
    
    with open("img_y_train.pkl", "rb") as fh:
        data = pickle.load(fh)
    y_train = np.array(pd.DataFrame(data)["label"].values.astype(np.float32)).flatten()
    
    with open("img_y_test.pkl", "rb") as fh:
        data = pickle.load(fh)
    y_test = np.array(pd.DataFrame(data)["label"].values.astype(np.float32)).flatten()
    

    y_test[y_test == 2] = -1
    y_test[y_test == 1] = 2
    y_test[y_test == -1] = 1
    
    y_train[y_train == 2] = -1
    y_train[y_train == 1] = 2
    y_train[y_train == -1] = 1
    

    X_train = []
    X_test = []
    
    for i in range(len(X_train_)):
        X_train.append(X_train_.values[i])
        
    for i in range(len(X_test_)):
        X_test.append(X_test_.values[i])
    
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    
    acc = []
    f1 = []
    precision = []
    recall = []
    seeds = random.sample(range(1, 200), 5)
    for seed in seeds:
        reset_random_seeds(seed)
        model = Sequential()
        model.add(Conv2D(100, (3, 3),  activation='relu', input_shape=(72, 72, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(50, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(3, activation = "softmax"))
        
        
        model.compile(Adam(learning_rate = 0.001), "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])
        
        model.summary()
        
    
        history = model.fit(X_train, y_train, epochs=50, batch_size=32,validation_split=0.1, verbose=1) 
        
        score = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        acc.append(score[1])
        
        test_predictions = model.predict(X_test)
        test_label = to_categorical(y_test,3)

        true_label= np.argmax(test_label, axis =1)

        predicted_label= np.argmax(test_predictions, axis =1)
        
        cr = classification_report(true_label, predicted_label, output_dict=True)
        precision.append(cr["macro avg"]["precision"])
        recall.append(cr["macro avg"]["recall"])
        f1.append(cr["macro avg"]["f1-score"])
    
    print("Avg accuracy: " + str(np.array(acc).mean()))
    print("Avg precision: " + str(np.array(precision).mean()))
    print("Avg recall: " + str(np.array(recall).mean()))
    print("Avg f1: " + str(np.array(f1).mean()))
    print("Std accuracy: " + str(np.array(acc).std()))
    print("Std precision: " + str(np.array(precision).std()))
    print("Std recall: " + str(np.array(recall).std()))
    print("Std f1: " + str(np.array(f1).std()))
    print(acc)
    print(precision)
    print(recall)
    print(f1)
    

    
if __name__ == '__main__':
    main()
    
    
