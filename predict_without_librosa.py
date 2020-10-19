import numpy as np
import glob
import os
#import tensorflow as tf
import collect
import train
import process_without_librosa
import matplotlib.pyplot as plt
import pyaudio
import librosa

def predict(f, wh, bh, wo, bo):
    ##### validation
    # Phase 1
    zh_p = np.dot([f], wh) + bh
    #print(zh_p)
    ah_p = train.sigmoid(zh_p)
    #print(ah_p)
    # Phase 2
    zo_p = np.dot(ah_p, wo) + bo
    #print(zo_p)
    ao_p = train.softmax(zo_p)
    #print(ao_p)
    return ao_p
 

model = np.load("model-mfccs-rasp/life-7-5.npz")
wh, bh, wo, bo = model['wh'], model['bh'], model['wo'], model['bo']

p = pyaudio.PyAudio()

#player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)
stream = p.open(format=collect.sample_format, channels=1, 
        rate=collect.RATE, input=True, frames_per_buffer=collect.CHUNK)

def predict_mfccs(volume_threshold, time_seconds):
    block = []
    b = 0
    html = ""
    for _ in range(int(time_seconds*collect.RATE/collect.CHUNK)): #do this for 10 seconds
        data=np.frombuffer(stream.read(collect.CHUNK),dtype=np.int16)
        #print(b)
        #print(len(block))
        #print(np.square(data[0:10].astype(np.float32)))
        volume = np.sqrt(
            np.mean(
                np.square(
                    data.astype(np.float32)*collect.SHORT_NORMALIZE)))
        #print (volume)
        #print('volume', volume)
        if(volume<volume_threshold and b==0):
            #not started
            continue
        elif(volume<volume_threshold and len(block)>0):
            #finish up this block
            b += 1
            if(b==collect.n_chunks_per_block-1):
                b = 0
        if(volume>volume_threshold):
            #maintain this buffer
            b = 0

        #mfccs = librosa.feature.mfcc(y=data.astype(np.float32), 
        #        sr=collect.RATE, n_mfcc=process.n_mfcc)

        block.append(data)

        if(len(block)==collect.n_chunks_per_block):
            d = np.array(block.copy())
            block = block[1:]
            
            #print(d.shape)
            d = d.reshape([-1])

            mfccs = process_without_librosa.mfcc(d, 
                collect.RATE, process_without_librosa.n_mfcc)#=d.astype(np.float32)

            mfccs = mfccs/np.max(np.abs(mfccs))
            #print(mfccs.shape)
            #plt.imshow(mfccs.T)
            #plt.show()
            #print(d.shape)
            ao = predict(mfccs.reshape([-1]), wh, bh, wo, bo)
            #print(ao)
            pcn = train.class_names[int(np.argmax(ao, axis=1))]
            #print(ao.shape)
            html_class = ""
            if(volume-volume_threshold<0.1):
                html_class = "small"
                print ("\033[96m{}\033[0m".format(pcn), end=" ", flush=True)
            elif(volume-volume_threshold<0.15):
                html_class = "small-medium"
                print ("\033[36m{}\033[0m".format(pcn), end=" ", flush=True)
            elif volume-volume_threshold<0.25:
                html_class = "medium"
                print ("\033[93m\033[1m{}\033[0m\033[0m".format(pcn), end=" ", flush=True)
            else :
                html_class = "large"
                print( "\033[91m\033[1m{}\033[0m\033[0m".format(pcn), end=" ", flush=True)
            currHTML = "<span class='{}'>{}</span>".format(html_class, 
                pcn)
            html += currHTML
            #print(currHTML)
    print()
    return html

def terminate_stream():
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ =="__main__":
    predict_mfccs(3e-6, 20)
    terminate_stream()