import pyaudio
import numpy as np
import struct
import random
import string
import wave

RATE    = 44100 #per second
CHUNK   = 1024*2 
#total_length = 10 #seconds
chunk_time = 1/RATE*CHUNK #seconds
top_n_freq = 20
n_chunks_per_block = 10

SHORT_NORMALIZE = (1.0/32768.0)
sample_format = pyaudio.paInt16

def buffer_to_freq(data):
    sp = abs(np.fft.fft(data))
    freq = np.fft.fftfreq(CHUNK)
    freq = abs(freq*RATE)
    #f=[]
    temp = np.argpartition(-sp, top_n_freq)
    result_indices = temp[:top_n_freq]
    f = freq[result_indices]
    return f


def main(volume_threshold, class_name, sound_type, n_entries):
    #t = np.linspace(0, chunk_time, CHUNK)

    print("chunk_time", chunk_time, "s")
    np.set_printoptions(threshold=CHUNK)

    p = pyaudio.PyAudio()

    #player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

    read_in = []
    block = []
    i = 0
    b = 0
    while i<n_entries:
        data=np.frombuffer(stream.read(CHUNK),dtype=np.int16)
        volume = np.sqrt(
            np.mean(
                np.square(
                    data.astype(np.float32)*SHORT_NORMALIZE)))
        #print(b)

        if(volume<volume_threshold and b==0):
            #not started
            continue
        elif(volume<volume_threshold and len(block)>0):
            #finish up this block
            b += 1
            if(b==n_chunks_per_block):
                b = 0
        if(volume>volume_threshold):
            #maintain this buffer
            b = 0

        f=buffer_to_freq(data)
        block.append(f)

        if(len(block)==n_chunks_per_block):
            read_in.append(block.copy())
            block = block[1:]
            print("{}: {}/{}".format(class_name, i, n_entries))
            i+=1
        
    read_in = np.array(read_in)
    print(len(read_in), len(read_in[0]) , len(read_in[0, 0]))
    #print(read_in)
    np.savez("./data/{}-{}-{}-{}.npz".format(sound_type, class_name, n_entries,
    ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))), data = read_in)

    stream.stop_stream()
    stream.close()
    p.terminate()

def record( volume_threshold, class_name, sound_type, n_entries ):

    print("chunk_time", chunk_time, "s")
    np.set_printoptions(threshold=CHUNK)

    p = pyaudio.PyAudio()

    #player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)
    stream = p.open(format=sample_format, channels=1, rate=RATE, 
                    input=True, frames_per_buffer=CHUNK)

    read_in = []
    i=0
    b=0
    block = 0
    while i<n_entries:
        data=np.frombuffer(stream.read(CHUNK),dtype=np.int16)
        volume = np.sqrt(
            np.mean(
                np.square(
                    data.astype(np.float32)*SHORT_NORMALIZE)))
        #print(b)

        if(volume<volume_threshold and b==0):
            #not started
            continue
        elif(volume<volume_threshold and len(block)>0):
            #finish up this block
            b += 1
            if(b==n_chunks_per_block):
                b = 0
        if(volume>volume_threshold):
            #maintain this buffer
            b = 0
        
        #i+=1
        
        block+=1
        
        read_in.append(data)
        if(block==n_chunks_per_block):
            block-=1
            print("{}: {}/{}".format(class_name, i, n_entries))
            i+=1
    #read_in = np.array(read_in)
    #print(len(read_in), len(read_in[0]) , len(read_in[0, 0]))

    filename = "./wave/{}-{}-{}-{}.wav".format(sound_type, class_name, n_entries,
    ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)))

    #print(read_in)
    #np.savez(filename, data = read_in)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(read_in))
    wf.close()

if __name__ =="__main__":
    ##############
    # Before running this script, create 'data' folder 
    # in the same folder as this script 
    ##############
    #threshold to record, sound type, test/train, num of clips to record
    #main(0.10,            "punch",   "test",     100)
    record(0.10, "knock-cathy", "train", 100)