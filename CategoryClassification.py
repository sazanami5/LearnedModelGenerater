import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

tf.random.set_seed(11)
batch_size= 100
epochs= 20
labelColumn= "あなたの高校時代の芸術選択科目を教えてください。"
trainRatio = 0.7
path = "data2/dataset.csv"
num_classes=  4

#init
accuracyRatio=0

def createDataset(path:str, trainRatio:float,labelColumn)-> list:
    #load data
    dataset = pd.read_csv(path, encoding= "utf-8")
    
    # object型をcategory型に変換
    for column in dataset.columns:
        dataset[column] = pd.Categorical(dataset[column])
        dataset[column] = dataset[column].cat.codes
        
    # create label dataset
    # convert class vectors to binary class matrices
    y_dataset = dataset.pop(labelColumn)
    y_dataset = y_dataset.values
    y_dataset = tf.keras.utils.to_categorical(y_dataset, num_classes)
    
    #create explanatory variable
    x_dataset = dataset.values
    
    datasetLength= len(dataset)
    x_train = x_dataset[0:math.floor(datasetLength*trainRatio)]
    y_train = y_dataset[0:math.floor(datasetLength*trainRatio)]
    x_test = x_dataset[math.floor(datasetLength*trainRatio):]
    y_test = y_dataset[math.floor(datasetLength*trainRatio):]
    print(len(x_train[0]))
    return x_train , y_train , x_test, y_test, len(x_train[0])

def createCompiledModel(train_columns):
    # モデル作成
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(train_columns,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy'])
    return model

def trainDataset(x_train, y_train, x_test, y_test, batch_size , epochs, train_columns):
    model= createCompiledModel(train_columns)
    # モデルを学習
    history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    
    # テストデータに対して誤差と精度を評価
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score, history
    
def plotTrain(history)->None:
    # visualize the learning (学習の状況を可視化)
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    fig =plt.figure()
    plt.plot(range(len(loss)), loss, label='loss')
    plt.plot(range(len(acc)), acc, label='acc')
    plt.plot(range(len(val_loss)), val_loss, label='val_loss',linestyle="--")
    plt.plot(range(len(val_acc)), val_acc, label='val_acc',linestyle="--")
    plt.xlabel('epochs')
    plt.ylabel('accuracy and loss')
    plt.show
    
#%%
[x_train , y_train , x_test, y_test, train_columns] = createDataset(path, trainRatio,labelColumn)
score, history = trainDataset(x_train , y_train , x_test, y_test,batch_size, epochs, train_columns)
plotTrain(history)