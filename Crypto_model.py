
#imports
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import deque
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, LSTM, BatchNormalization)
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

#Global variables
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
EPOCHS = 10
BATCH_SIZE = 64

#Create unique name for each model parameter to represent them accurately and distinctly
NAME = f'{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}'

#----------------------------------------------------------------------------#
def classify(current, future):
    '''
    this function will compare future and current volume values to determine
    whether the volume increased or decreased over the given interval.
    IN: current volume, future volume
    OUT: int(0/1) volume increase (t/f)
    '''
    
    if float(future) > float(current):
        return 1
    else:
        return 0
#----------------------------------------------------------------------------#
def preprocess_dataset(dataset):
    
    '''
    this function will preprocess the data, creating a matrix of features (x)
    and a dependent variable vector (y).  X will contain all columns barring 'target',
    scaled to represent pct_change.  The target (volume increase y/n is represented with y).
    NOTE: If running a unique dataset, bar any values that do not require feature scaling
    and append them as a np array to the matrix of features after data preprocessing.
    NOTE PART 2: SEQ_LEN determines the length of the sequence.  Will only return
    that many values.
    
    IN: Dataset 
    OUT: matrix of features, dependent variable vector.
    '''
    
    #Would obviously ruin the model to include the future volume in the dataset.
    dataset = dataset.drop('future', 1) 
    
    #This loop is what normalizes the data with pct_change.  It goes through
    #each column, not each value.
    for col in dataset.columns:
        if col != 'target':
            #Normalize with pct change
            dataset[col] = dataset[col].pct_change()
            #drop NaN
            dataset.dropna(inplace = True)
            dataset[col] = preprocessing.scale(dataset[col].values)
    
    #NOTE: dropna removes any NaN values.
    #dropna again for any created by pct_change
    dataset.dropna(inplace = True)
    
    #Sequential data list
    sequential_data = []
    
    #Create deque - when deque hits len = maxlen, will automatically pop old values to add new.
    prev_days = deque(maxlen = SEQ_LEN)
    
    for i in dataset.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            #append x & y - features & label
            sequential_data.append([np.array(prev_days), i[-1]])
        
    random.shuffle(sequential_data)
    
    
    for seq, target in sequential_data:
        if target == 0:
            volume_decrease.append([seq, target])
        elif target == 1:
            volume_increase.append([seq, target])
          
    #Identify which list has fewer values, make lower == len(smallest)
    lower = min(len(volume_increase), len(volume_decrease))
        
    #Equalize Volume_Increase and Volume_decrease to avoid creating too much bias in the model.
    #Note: target is not computed with any advanced math, it is simply telling whether
    #The future volume will be higher than the current volume.  Based on the logic
    # in the function classify.
    volume_decrease = volume_increase[:lower]
    volume_increase = volume_decrease[:lower]
        
    #Shuffle again so the data isn't split 50/50
    random.shuffle(sequential_data)
        
    #X contains the matrix of features, y contains the dependent variable vector.
        
    x = []
    y = []
        
    #Sequence contains matrix of features, scaled to pct change, target contains
    #Is target (0/1)
    for sequence, target in sequential_data:
        x.append(sequence)
        y.append(target)
            
            
    return np.array(x), y
#----------------------------------------------------------------------------#
'''Pre-training'''
    
#Initialize main dataset (or dataframe) - will contain final values (prior to preprocessing)
main_dataset = pd.DataFrame()
#Keep track of ratio name
ratioName = ['KMD_USD']

for ratio in ratioName:
    datalink = f'{ratio}.csv'
    dataset = pd.read_csv('volume_history.csv')
    dataset.set_index("date", inplace = True) #inplace = edit original object
    
    if(len(main_dataset) == 0):
        main_dataset = dataset
    else:
        main_dataset = main_dataset.join(dataset)
    
main_dataset['future'] = main_dataset['close'].shift(+FUTURE_PERIOD_PREDICT)
main_dataset['target'] = list(map(classify, main_dataset['close'], main_dataset['future']))

times = sorted(main_dataset.index.values)
last_5pct = times[-int(0.05*len(times))]

#Separate validation and training data
validation_main_dataset = main_dataset[(main_dataset.index >= last_5pct)]
main_dataset = pd.DataFrame(data = main_dataset[main_dataset.index < last_5pct])


x_train, y_train = preprocess_dataset(main_dataset)
x_validate, y_validate = preprocess_dataset(validation_main_dataset)

#----------------------------------------------------------------------------#
'''Model training'''

model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))
opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)

#COMPILE MODEL
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt,
              metrics = ['accuracy'])


tensorboard = TensorBoard(log_dir = f'logs/{NAME}')
#Checkpoint object - copy and pasted
filepath = "RNN_Final-{epoch:02d}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy',
                        verbose=1, save_best_only=True, mode='max')) # saves only the best one


#Avoiding ValueError raised if training / validation sets contain lists instead of np array
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_validate = np.asarray(x_validate)
y_validate = np.asarray(y_validate)



history = model.fit(x_train, y_train,
                    batch_size = BATCH_SIZE,
                        epochs = EPOCHS,
                        validation_data= (x_validate, y_validate),
                        callbacks=[tensorboard, checkpoint])
