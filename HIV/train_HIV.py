import tensorflow as tf
from itertools import combinations_with_replacement
from tensorflow.keras import models, layers



def train(dataset, Num_layer, dim):
    X_train, X_test, y_train, y_test = dataset
    
    temp=[]
    for i in range(1,Num_layer+1):
        temp+=list(combinations_with_replacement(dim,i))

    BestAcc=0

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    
    for i in temp:
        model = models.Sequential()
        for j in i:
            model.add(layers.Dense(j, activation='relu'))
        model.add(layers.Dense(1, activation = 'sigmoid'))
    
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, verbose=1, validation_split = 0.1, callbacks=[early_stop])
    
        test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=1)
        if test_acc > BestAcc:
            Finalmodel = model
            BestAcc=test_acc
            BestModelArchitecture=i
        del model
    print("<Best performing architecture>\nLayers No:",len(BestModelArchitecture),"\nDimensions:",BestModelArchitecture)

    return Finalmodel
