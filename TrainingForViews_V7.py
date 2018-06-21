# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:18:03 2018

@author: Adrian
"""
import tensorflow as tf
import os
import numpy as np
import gzip
from itertools import cycle
from skimage.transform import resize
import random


label_path=os.path.join(os.getcwd(), 'train_labels.npy.gz')
train_path=os.path.join(os.getcwd(), 'train_data')
model_path='View_Custom'

batch_size=10
vid_length=50

total_numbers=len(os.listdir(train_path))
train_numbers=10000
val_numbers=total_numbers-train_numbers

train_bag=list(range(train_numbers))
evaluation_bag=list(range(train_numbers+1, total_numbers))

def shuffling():
    global iter_train, iter_val
    random.shuffle(train_bag)
    random.shuffle(evaluation_bag)
    iter_train=cycle(train_bag)
    iter_val=cycle(evaluation_bag)
    print('shuffle done')
    
shuffling()

with gzip.GzipFile('train_labels.npy.gz', "r") as gf:
    labels=np.load(gf)


def regrid(img, x=600, y=800):
    img_resized=resize(img, (x,y))
    return img_resized


def fetch_data(batch_size, train=True, generator=False):
    file_list=[]
    k=lx=sa=c4=c2=0
    bag=[]
    for i in range(batch_size):
        if train: iters=iter_train
        else: iters=iter_val
        
        
        file='{}.npy.gz'
        tar_file=next(iters)
        
        if np.argmax(labels[tar_file])==4: k+=1
        elif np.argmax(labels[tar_file])==3: c2+=1
        elif np.argmax(labels[tar_file])==2: c4+=1
        elif np.argmax(labels[tar_file])==1: sa+=1
        elif np.argmax(labels[tar_file])==0: lx+=1
        while k>batch_size/5:
            tar_file=next(iters)
            if np.argmax(labels[tar_file]) not in (4,): 
                break      
        
        
        
        with gzip.GzipFile(os.path.join(train_path, file.format(tar_file)), "r") as gf:
            tar_array=np.load(gf)
        while len(tar_array.shape) <4:
            tar_file=next(iters)
            with gzip.GzipFile(os.path.join(train_path, file.format(tar_file)), "r") as gf:
                tar_array=np.load(gf)
        
        bag.append(np.argmax(labels[tar_file]))
        boo = tar_array.shape==(50,600,800,3)
        
        if not boo:
            print(tar_file, tar_array.shape)
            for j, img in enumerate(tar_array):
                img=regrid(img)
                img=np.expand_dims(img, axis=0)
                if j==0: tar_array_sub=img
                else: tar_array_sub=np.concatenate((tar_array_sub, img))
            tar_array=tar_array_sub

        #tar_array=tar_array[:30]
        
        file_list.append(tar_file)
        tar_array=np.expand_dims(tar_array, axis=0)
        #print(tar_array.shape)
        if i==0:
            train_batch=tar_array
        else:
            train_batch=np.concatenate((train_batch, tar_array))
        
        #print('This is iterfile: {}'.format(tar_file))  
    
    train_batch=train_batch.astype('float32')/255
    label_batch=labels[file_list]
    print(bag)

    if generator==False:
        return train_batch, label_batch
    
    else:
        final_list=[]
        for i in range(len(train_batch)):
            final_list.append((np.expand_dims(train_batch[i], axis=0), np.expand_dims(label_batch[i], axis=0)))
        final_list=cycle(tuple(final_list))
        
        return final_list
        



from keras.models import Model, model_from_json, Sequential
from keras.layers import TimeDistributed, Dense, SimpleRNN, Input, Flatten, Convolution2D, InputLayer, Reshape, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam






def made_model():
    model=Sequential()
    
    model.add(InputLayer(input_shape=(50, 600, 800, 3)))
    model.add(Reshape((5, 10, 600, 800, 3)))
    model.add(TimeDistributed(TimeDistributed(MaxPooling2D(pool_size=2, strides=2))))

    model.add(TimeDistributed(TimeDistributed(Conv2D(kernel_size=3, strides=1, filters=5, padding='same',
                     activation='relu', name='layer_conv1'))))
    model.add(TimeDistributed(TimeDistributed(MaxPooling2D(pool_size=2, strides=2))))
    model.add(TimeDistributed(TimeDistributed(Dropout(0.5))))
    model.add(TimeDistributed(TimeDistributed(BatchNormalization())))

    model.add(TimeDistributed(TimeDistributed(Conv2D(kernel_size=5, strides=1, filters=20, padding='same',
                     activation='relu', name='layer_conv2'))))
    model.add(TimeDistributed(TimeDistributed(MaxPooling2D(pool_size=2, strides=2))))
    model.add(TimeDistributed(TimeDistributed(Dropout(0.5))))
    model.add(TimeDistributed(TimeDistributed(BatchNormalization())))

    model.add(TimeDistributed(TimeDistributed(Conv2D(kernel_size=5, strides=1, filters=20, padding='same',
                     activation='relu', name='layer_conv2'))))    
    model.add(TimeDistributed(TimeDistributed(MaxPooling2D(pool_size=2, strides=2))))
    model.add(TimeDistributed(TimeDistributed(Dropout(0.5))))

    model.add(TimeDistributed(TimeDistributed(Flatten())))
    model.add(TimeDistributed(TimeDistributed(Dense(256, activation='relu'))))
    
    model.add(TimeDistributed(SimpleRNN(256, return_sequences=False, stateful=False)))
    model.add(SimpleRNN(256))
    model.add(Dense(5, activation='softmax'))
    
    optimizer=Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model



def optimize(iter_n, samples=30, batch_size=5, validation_split=0.0):
    best=0
    for i in range(iter_n):
        print('This is {}th loop'.format(i+1))
        x,y=fetch_data(samples)
        history=model.fit(x, y, epochs=2, validation_split=validation_split, batch_size=batch_size)
        
        if (i+1)%10==0:
            acc=evaluation_test()
            acc=acc[1]            
            if acc>best:    
                best=acc
                model.save_weights(model_path)
                print('The {} loop saved with validaction accuracy: {}'.format(i+1, acc))
        
        if(i+1)%100==0:
            shuffling()
            
        
    #res=evaluation_test(100)
    #print(res)
    return history


def optimize_gen(iter_n, samples=20):
    
    for i in range(iter_n):
        print('This is {}th loop'.format(i+1))
        
        history=model.fit_generator(fetch_data(samples, True, True), epochs=5, steps_per_epoch=samples)
        
        if (i+1)%10==0:
            
            model_saving(model_path)
            print('The {} loop saved'.format(i+1))
    #res=evaluation_test(100)
    #print(res)
    return history


def evaluation_test(x=100):
    n=x//20
    loss=accuracy=0
    for i in range(n):
        x,y=fetch_data(20, train=False)
        res=model.evaluate(x,y, batch_size=2)

        loss+=res[0]
        accuracy+=res[1]        
        
        
    return [loss/n, accuracy/n]
    
def model_saving(x):
    model_json=model.to_json()
    with open(str(x)+'_arch.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(str(x)+'_weights.h5')
    print('model saved')

def model_loading(x):
    with open(str(x)+'_arch.json', 'r') as json_file:
        model_json=json_file.read()
    model=model_from_json(model_json)
    model.load_weights(str(x)+'_weights.h5')
    optimizer=Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print('model loaded')
    return model

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


model=made_model()

try: 
    model.load_weights(model_path)
    print('Model reused')
except:
    print('New model')