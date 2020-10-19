#import librosa
import numpy as np
import glob
import os
#import tensorflow as tf
import collect
import matplotlib.pyplot as plt
import random
import string
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window


hop_length = 512
n_mfcc = 20
FFT_size = 2048
mfcc_len = int(np.ceil((collect.CHUNK*collect.n_chunks_per_block)/hop_length))+1
#int((int(collect.CHUNK*(100+collect.n_chunks_per_block-1)) - FFT_size) / hop_length) + 1
#
#class_names = ["ahh", "clap", "jingle", "knock", "water", "punch"]
#class_names = ["ahh-david", "awe-david", "eee-david", "wu-david", "ae-david"]
#class_names = ["ahh-cathy", "awe-cathy", "eee-cathy", "wu-cathy", "ae-cathy"]
class_names = ["ahh-cathy", "clap-cathy", "knock-cathy", "water-cathy", "jingle-cathy" ]

def frame_audio(audio, FFT_size=2048, hop_size=512, sample_rate=44100):
    # hop_size not in ms, in samples
    
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = hop_size #np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num,FFT_size))
    
    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
    
    return frames

def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    #print("MEL min: {0}".format(fmin_mel))
    #print("MEL max: {0}".format(fmax_mel))
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters

def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis

def mfcc(audio, sample_rate, n_mfcc):
    audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_length, sample_rate=sample_rate)
    #print("audio framed", audio_framed.shape)
    window = get_window("hann", FFT_size, fftbins=True)
    audio_win = audio_framed * window
    audio_winT = np.transpose(audio_win)

    audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
    
    freq_min = 0
    freq_high = sample_rate / 2
    mel_filter_num = n_mfcc
    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
    audio_fft = np.transpose(audio_fft)
    #print("audio_fft", audio_fft.shape)
    audio_power = np.square(np.abs(audio_fft))
    #print("audio_power", audio_power.shape)

    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate)
    filters = get_filters(filter_points, FFT_size)

    # taken from the librosa library
    enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
    filters *= enorm[:, np.newaxis]

    audio_filtered = np.dot(filters, np.transpose(audio_power))
    audio_log = 10.0 * np.log10(audio_filtered)

    dct_filter_num = n_mfcc

    dct_filters = dct(dct_filter_num, mel_filter_num)

    cepstral_coefficents = np.dot(dct_filters, audio_log)
    mfccs = cepstral_coefficents/np.max(np.abs(cepstral_coefficents))
    return mfccs


def get_mfccs(cn, data_type):
    files = glob.glob('./rasp/{}-{}-*.wav'.format(data_type, cn))
    data_set = np.empty([1, n_mfcc, mfcc_len])
    #label_set = np.empty([len(class_names)])

    for f in files:
        try:
            sample_rate, audio = wavfile.read(f) 
            #assert sample_rate == collect.RATE
            audio = np.asarray(audio)
            print("entire audio shape", audio.shape)
            #print(np.max(audio))
            #https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
            audio = audio/np.max(audio)
            
            for i in range (int(np.ceil(audio.shape[0]/collect.CHUNK))-collect.n_chunks_per_block):
                #print(i, i*collect.CHUNK, 
                #    collect.CHUNK*i+int(np.ceil(collect.CHUNK*collect.n_chunks_per_block)))
            #    mfccs = librosa.feature.mfcc(
            #        y=audio[i*collect.CHUNK:
            #        collect.CHUNK*i+int(np.ceil(collect.CHUNK*collect.n_chunks_per_block))], 
            #            sr=sample_rate, n_mfcc=n_mfcc)
                mfccs = mfcc(audio[i*collect.CHUNK:
                    collect.CHUNK*i+int(np.ceil(collect.CHUNK*collect.n_chunks_per_block))],
                    sample_rate, n_mfcc)
                print(mfccs.shape)
                mfccs = mfccs/np.max(np.abs(mfccs))
                #print("where does it go?", mfccs.shape)
            #    if(mfccs.shape[1]==mfcc_len):
                data_set = np.vstack([data_set, [mfccs]])
                #print("where does my data set go?", data_set)

        except IOError:
            pass
    
    data_set = data_set[1:]
    num_filters = data_set.shape[0]

    max_num_plots_per_figure = 20    
    total_num_figures = int(np.ceil(num_filters/float(max_num_plots_per_figure)))
    #print("total_num_figures (rows)", total_num_figures)

    fig = plt.figure()
    for fig_ind in range(total_num_figures):
        #print(fig_ind)
        start_filter_to_show = fig_ind * max_num_plots_per_figure
        end_filter_to_show   = min(num_filters, start_filter_to_show + max_num_plots_per_figure)

        filters_to_show = list(range(start_filter_to_show,end_filter_to_show))
        #print("columns", len(filters_to_show))

        for k, filter_ind in enumerate(filters_to_show):
            #print(fig_ind, k, filter_ind)
            plt.subplot(total_num_figures, max_num_plots_per_figure, fig_ind*max_num_plots_per_figure+k+1); #plt.title('filter %d' %(filter_ind))
            im = plt.imshow(data_set[filter_ind, :,:].T,cmap='jet', 
                    interpolation="none", aspect=1) #0.1
            plt.axis('off')
        
    plt.tight_layout()
    fig.suptitle(cn+' mfccs', fontsize=16)
    #plt.show()
    plt.savefig('mfccs-{}-{}-rasp.png'.format(data_type, cn))

    return data_set


if __name__ == "__main__":
    #get_mfccs("clap", "test")
    #get_mfccs("jingle", "test")
    #get_mfccs("ahh", "test")
    #get_mfccs("eee", "test")
    for data_type in ["test", "train"]:
        for cn in class_names:
            print(data_type, cn)
            data_set = get_mfccs(cn, data_type)
            np.savez("./data-mfccs-rasp/{}-{}-y.npz".format(data_type, cn), data = data_set)
