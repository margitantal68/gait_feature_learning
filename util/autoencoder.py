# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import util.settings as stt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn import preprocessing


def build_FCN_autoencoder( input_shape, num_epochs, file_path, fcn_filters=128):
    input_dim = stt.DIMENSIONS
    input_layer = keras.layers.Input(shape=input_shape)
    conv1 = keras.layers.Conv1D(filters = fcn_filters, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters = 2*fcn_filters, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    
    conv3 = keras.layers.Conv1D(filters = fcn_filters, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    
    encoded = keras.layers.GlobalAveragePooling1D()(conv3)
    dim_encoded = K.int_shape(encoded)[1]
  

    # DECODER
    # stt.FEATURES must be a multiple of 128
    factor = 1
    if( dim_encoded < stt.FEATURES ):
        factor = (int) (stt.FEATURES / dim_encoded)
  
    h = keras.layers.Reshape((dim_encoded, 1) )(encoded)
    h = keras.layers.UpSampling1D( factor )(h)  
    conv3 = keras.layers.Conv1D( filters = fcn_filters, kernel_size=3, padding='same')(h)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    conv2 = keras.layers.Conv1D( filters = 2 * fcn_filters, kernel_size=5, padding='same')(conv3)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv1 = keras.layers.Conv1D(filters = input_dim, kernel_size=8, padding='same')(conv2)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)
    decoded = conv1

    encoder = keras.Model(input_layer, encoded)
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, 
                                                  min_lr=0.0001) 
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True, verbose=1)
    callbacks = [reduce_lr, model_checkpoint]
    # pr
def save_and_plot_history( history, model_name ):
    model_name_without_ext = model_name[0 : model_name.find('.')]
    print(model_name_without_ext)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Autoencoder Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig(stt.TRAINING_CURVES_PATH+'/' + model_name+'_' + '.png', format='png')

# TRAIN and SAVE autoencoder
# Parameters:
#   df - dataframe 
#   model_name - e.g. 'autoencoder_FCN.h5'
#   num_epoch - number of epochs
#   fcn_filters - used only for FCN models

def train_autoencoder(df, model_name, num_epochs=stt.EPOCHS, fcn_filters=128):
    # filepath = stt.TRAINED_MODELS_PATH + "/" + model_name
    # build autoencoder
    input_shape = (stt.FEATURES, stt.DIMENSIONS)
    file_path = stt.TRAINED_MODELS_PATH + "/" + model_name
    callbacks, encoder, model = build_FCN_autoencoder( input_shape, num_epochs, file_path, fcn_filters = fcn_filters)

    # split dataframe
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y = enc.transform(y.reshape(-1, 1)).toarray()
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)
 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=stt.RANDOM_STATE)
    mini_batch_size = int(min(X_train.shape[0]/10, stt.BATCH_SIZE))

    # convert to tensorflow dataset
    X_train = np.asarray(X_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, X_val))

    BATCH_SIZE = mini_batch_size
    SHUFFLE_BUFFER_SIZE = 100

    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    start_time = time.time()
    # train model
    history = model.fit(train_ds,
                              epochs=num_epochs,
                              shuffle=False,
                              validation_data=val_ds,
                              callbacks=callbacks)
    
    duration = time.time() - start_time
    print("Training duration: "+str(duration/60))                        
    
    save_and_plot_history( history, model_name )
    model = tf.keras.models.load_model(file_path)
    
    num_layers = len(model.layers)
    half = num_layers//2
    feat_model = tf.keras.Sequential()
    for layer in model.layers[0:half]:
        feat_model.add(layer)
    # feat_model.summary()
    encoder_name = 'encoder_' + model_name
    # print('Saved encoder: ' + encoder_name)
    feat_model.save(stt.TRAINED_MODELS_PATH + '/' + encoder_name)


# Use a pretrained model for feature extraction
# Load the encoder
def get_autoencoder_output_features( df, model_name ):
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)

    model_path = stt.TRAINED_MODELS_PATH + '/' + model_name
   
    print(model_path)
    model = tf.keras.models.load_model(model_path)
    # model.summary()
    X = np.asarray(X).astype(np.float32)

    features = model.predict( X )
    df = pd.DataFrame( features )
    df['user'] = y 
    return df

# Use a pretrained model for samples generation
# df - input dataset
# model_name -  autoencoder
# return the generated data
def generate_autoencoder_samples( df, model_name ):
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)

    model_path = stt.TRAINED_MODELS_PATH + '/' + model_name
   
    model = tf.keras.models.load_model(model_path)
    # model.summary()
    X = np.asarray(X).astype(np.float32)
    generated_samples = model.predict( X )
    generated_samples = generated_samples.reshape(-1, stt.FEATURES * stt.DIMENSIONS)
    df = pd.DataFrame( generated_samples )
    df['user'] = y 
    return df






def save_and_plot_history( history, model_name ):
    model_name_without_ext = model_name[0 : model_name.find('.')]
    print(model_name_without_ext)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Autoencoder Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig(stt.TRAINING_CURVES_PATH+'/' + model_name+'_' + '.png', format='png')