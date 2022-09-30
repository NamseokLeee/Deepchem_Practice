import tensorflow as tf
from itertools import combinations_with_replacement
from tensorflow.keras import models, layers



def train(dataset, Num_layer, dim):
    
    train_dataset, test_dataset = dataset
    
    a= train_dataset.X.astype(float)
    b= test_dataset.X.astype(float)
    
    temp=[]
    for i in range(1,Num_layer+1):
        temp+=list(combinations_with_replacement(dim,i))

    BestMSE=0

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    for i in temp:
        model = models.Sequential()
        for j in i:
            model.add(layers.Dense(j, activation='relu'))
        model.add(layers.Dense(1))
    
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        history = model.fit(a, train_dataset.y, epochs=100, verbose=1, validation_split = 0.1, callbacks=[early_stop])
        loss, mae, mse = model.evaluate(b, test_dataset.y, verbose=2)
        if mse > BestMSE:
            Finalmodel = model
            Finalhistory = history
            BestMSE = mse
            BestModelArchitecture=i
        del model
    print("<Best performing architecture>\nLayers No:",len(BestModelArchitecture),"\nDimensions:",BestModelArchitecture)
    
    return Finalmodel, Finalhistory