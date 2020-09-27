import numpy as np
import glob
import os
#import tensorflow as tf
import collect
import matplotlib.pyplot as plt

class_names = ["ahh", "clap", "jingle", "knock", "water", "punch"]#,  "scratch", 

def plot_data(cn):
    files = glob.glob('./data/test-'+cn+'*.npz')

    train_data_set = np.empty([1, collect.n_chunks_per_block, collect.top_n_freq])

    for f in files:
        try:
            fn = os.path.basename(f)
            print(fn)
            first_dash = fn.find("-")
            data_type = fn[0:first_dash]
            print(data_type)
            second_dash = fn.find("-", first_dash+1)
            cn = fn[first_dash+1: second_dash]
            print(cn)
            if(cn in class_names):
                #c_index = class_names.index(cn)
                data = np.load(f)
                d = data['data']
                print(d.shape)
                #d = d.reshape(d.shape[0], -1)
                #one_hot = np.zeros([d.shape[0], len(class_names)])
                #one_hot[:, c_index]=1
                
                train_data_set = np.vstack([train_data_set, d])
                #test_label_set = np.vstack([test_label_set, one_hot])
                
        except IndexError:
            print("probably not our data file")
            pass

    train_data_set = train_data_set[1:]
    num_filters = train_data_set.shape[0]
        
    ylims = np.array([-0.01, 1.01]) * 10000#np.max(train_data_set)
    print(cn, "max", np.max(train_data_set))

    max_num_plots_per_figure = 20   
    total_num_figures = int(np.ceil(num_filters/float(max_num_plots_per_figure)))

    fig = plt.figure()
    for fig_ind in range(total_num_figures):
        start_filter_to_show = fig_ind * max_num_plots_per_figure
        end_filter_to_show   = min(num_filters, start_filter_to_show + max_num_plots_per_figure)

        filters_to_show = list(range(start_filter_to_show,end_filter_to_show))

        for k, filter_ind in enumerate(filters_to_show):
            #print(fig_ind, k, filter_ind)
            plt.subplot(total_num_figures, len(filters_to_show),fig_ind*max_num_plots_per_figure+k+1); #plt.title('filter %d' %(filter_ind))
            im = plt.imshow(train_data_set[filter_ind, :,:].T,cmap='jet', 
                    interpolation="none", clim=ylims, aspect=1) #0.1
            plt.axis('off')
        
    plt.tight_layout()
    fig.suptitle(cn+' features', fontsize=16)
    #plt.show()
    plt.savefig("features-"+cn+'.png')

if __name__ == "__main__":
    plot_data("ahh")
    #plot_data("clap")
    #plot_data("water")
    #plot_data("knock")
    #plot_data("jingle")