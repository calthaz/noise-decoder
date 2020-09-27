import librosa
import numpy as np
import glob
import os
#import tensorflow as tf
import collect
import matplotlib.pyplot as plt
import random
import string

hop_length = 512
n_mfcc = 20
mfcc_len = int(np.ceil(collect.CHUNK*collect.n_chunks_per_block/hop_length))+1
#class_names = ["ahh", "clap", "jingle", "knock", "water", "punch"]
#class_names = ["ahh-david", "awe-david", "eee-david", "wu-david", "ae-david"]
#class_names = ["ahh-cathy", "awe-cathy", "eee-cathy", "wu-cathy", "ae-cathy"]
class_names = ["ahh-cathy", "clap-cathy", "knock-cathy", "water-cathy", "jingle-cathy" ]

def get_mfccs(cn, data_type):
    files = glob.glob('./wave/{}-{}-*.wav'.format(data_type, cn))
    data_set = np.empty([1, n_mfcc, mfcc_len])
    #label_set = np.empty([len(class_names)])

    for f in files:
        try:
            audio, sample_rate = librosa.load(f, res_type='kaiser_fast', sr=None) 
            #assert sample_rate == collect.RATE
            audio = np.asarray(audio)
            print("audio shape", audio.shape)
            print(np.max(audio))
            for i in range (int(np.ceil(audio.shape[0]/collect.CHUNK))):
                #print(i, i*collect.CHUNK, 
                #    collect.CHUNK*i+int(np.ceil(collect.CHUNK*collect.n_chunks_per_block)))
                mfccs = librosa.feature.mfcc(
                    y=audio[i*collect.CHUNK:
                    collect.CHUNK*i+int(np.ceil(collect.CHUNK*collect.n_chunks_per_block))], 
                        sr=sample_rate, n_mfcc=n_mfcc)
                #print(mfccs)
                mfccs = mfccs/np.max(np.abs(mfccs))
                #print("where does it go?", mfccs.shape)
                if(mfccs.shape[1]==mfcc_len):
                    data_set = np.vstack([data_set, [mfccs]])
                #print("where does my data set go?", data_set)

        except IOError:
            pass
    
    data_set = data_set[1:]
    num_filters = data_set.shape[0]

    max_num_plots_per_figure = 20    
    total_num_figures = int(np.ceil(num_filters/float(max_num_plots_per_figure)))
    print("total_num_figures (rows)", total_num_figures)

    fig = plt.figure()
    for fig_ind in range(total_num_figures):
        #print(fig_ind)
        start_filter_to_show = fig_ind * max_num_plots_per_figure
        end_filter_to_show   = min(num_filters, start_filter_to_show + max_num_plots_per_figure)

        filters_to_show = list(range(start_filter_to_show,end_filter_to_show))
        print("columns", len(filters_to_show))

        for k, filter_ind in enumerate(filters_to_show):
            print(fig_ind, k, filter_ind)
            plt.subplot(total_num_figures, max_num_plots_per_figure, fig_ind*max_num_plots_per_figure+k+1); #plt.title('filter %d' %(filter_ind))
            im = plt.imshow(data_set[filter_ind, :,:].T,cmap='jet', 
                    interpolation="none", aspect=1) #0.1
            plt.axis('off')
        
    plt.tight_layout()
    fig.suptitle(cn+' mfccs', fontsize=16)
    #plt.show()
    plt.savefig('mfccs-{}-{}.png'.format(data_type, cn))

    return data_set


if __name__ == "__main__":
    #get_mfccs("clap", "test")
    #get_mfccs("jingle", "test")
    #get_mfccs("ahh", "test")
    get_mfccs("eee", "test")
    for data_type in ["test", "train"]:
        for cn in class_names:
            data_set = get_mfccs(cn, data_type)
            np.savez("./data-mfccs-no-res/{}-{}-y.npz".format(data_type, cn), data = data_set)
