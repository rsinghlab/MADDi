
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def main():
    
        #this is created in the genetic preprocess jupyter notebook
        X_train = pd.read_pickle("X_train_vcf.pkl")
        y_train = pd.read_pickle("y_train_vcf.pkl")

        X_test = pd.read_pickle("X_test_vcf.pkl")
        y_test = pd.read_pickle("y_test_vcf.pkl")


        acc = []
        f1 = []
        precision = []
        recall = []
        seeds = random.sample(range(1, 200), 5)

        for seed in seeds:
            reset_random_seeds(seed)
            model = Sequential()
            model.add(Dense(128, input_shape = (15965,), activation = "relu")) 
            model.add(Dropout(0.5))
            model.add(Dense(64, activation = "relu"))
            model.add(Dropout(0.5))

            model.add(Dense(32, activation = "relu"))
            model.add(Dropout(0.3))

            model.add(Dense(32, activation = "relu"))
            model.add(Dropout(0.3))


            model.add(Dense(3, activation = "softmax"))

            model.compile(Adam(learning_rate = 0.001), "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])


            history = model.fit(X_train, y_train,epochs=50,batch_size=32,validation_split = 0.1, verbose=1) 

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
    
