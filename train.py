import numpy as np
import glob
import os
#import tensorflow as tf
import collect
import process
import matplotlib.pyplot as plt

#class_names = ["ahh", "clap", "knock", "water", "jingle" ]#,  "scratch", , "punch"
#class_names = ["ahh-david", "awe-david", "eee-david", "wu-david", "ae-david"]
#class_names = ["ahh-cathy", "awe-cathy", "eee-cathy", "wu-cathy", "ae-cathy"]
#class_names = ["ahh-cathy", "awe-cathy", "eee-cathy", "wu-cathy", "ae-cathy"]

class_names = ["ahh-cathy", "clap-cathy", "knock-cathy", "water-cathy", "jingle-cathy" ]

def load_data(n_features):
    files = glob.glob('./data-mfccs-no-res/*.npz')
    print(files)

    train_data_set = np.empty([1, n_features])
    train_label_set = np.empty([len(class_names)])

    test_data_set = np.empty([1, n_features])
    test_label_set = np.empty([len(class_names)])

    for f in files:
        try:
            fn = os.path.basename(f)
            print(fn)
            first_dash = fn.find("-")
            data_type = fn[0:first_dash]
            print(data_type)
            second_dash = fn.find("-", first_dash+1)
            third_dash = fn.find("-", second_dash+1)
            cn = fn[first_dash+1: third_dash]
            print(cn)
            if(cn in class_names):
                c_index = class_names.index(cn)
                data = np.load(f)
                d = data['data']
                d = d.reshape(d.shape[0], -1)
                one_hot = np.zeros([d.shape[0], len(class_names)])
                one_hot[:, c_index]=1
                if(data_type=="test"):
                    test_data_set = np.vstack([test_data_set, d])
                    test_label_set = np.vstack([test_label_set, one_hot])
                    
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(d.T, alpha=0.5)
                    ax.set_xlabel("index")
                    ax.set_ylabel("normalized power")
                    ax.set_title(cn)
                    plt.savefig("test-{}-data.png".format(cn))
                    
                if(data_type=="train"):
                    train_data_set = np.vstack([train_data_set, d])
                    train_label_set = np.vstack([train_label_set, one_hot])
                
        except IndexError:
            print("probably not our data file")
            pass

    test_data_set = test_data_set[1:]#/(collect.RATE/4)
    train_data_set = train_data_set[1:]#/(collect.RATE/4)

    test_label_set = np.array(test_label_set[1:])
    train_label_set = np.array(train_label_set[1:])
    print("test_data", test_data_set.shape, test_label_set.shape)
    print("train_data", train_data_set.shape, train_label_set.shape)


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(test_data_set.T, alpha=0.5)
    ax.set_xlabel("index")
    ax.set_ylabel("normalized freq")
    ax.set_title("all normalized data")
    plt.savefig("all-test-data.png")

    return train_data_set, train_label_set, test_data_set, test_label_set

'''
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_data_set, train_label_set)
    ).shuffle(batch_size*10).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_data_set, test_label_set)
    ).batch(batch_size)
'''

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(x):
    expA = np.exp(x)
    return expA / expA.sum(axis=1, keepdims=True)

def train(hidden_nodes, train_data_set, train_label_set, test_data_set, test_label_set):

    instances = train_data_set.shape[0]
    attributes = train_data_set.shape[1]
    #hidden_nodes = 10
    output_labels = len(class_names)

    wh = np.random.rand(attributes,hidden_nodes)
    bh = np.random.randn(hidden_nodes)

    wo = np.random.rand(hidden_nodes,output_labels)
    bo = np.random.randn(output_labels)
    lr = 1e-4

    error_cost = []
    early_stop = 0
    patience = 3
    min_loss = np.infty

    for epoch in range(100000):
    ############# feedforward

        # Phase 1
        zh = np.dot(train_data_set, wh) + bh
        ah = sigmoid(zh)

        # Phase 2
        zo = np.dot(ah, wo) + bo
        ao = softmax(zo)
    ########## Back Propagation

    ########## Phase 1

        dcost_dzo = ao - train_label_set
        dzo_dwo = ah

        dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

        dcost_bo = dcost_dzo

    ########## Phases 2

        dzo_dah = wo
        dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
        dah_dzh = sigmoid_der(zh)
        dzh_dwh = train_data_set
        dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

        dcost_bh = dcost_dah * dah_dzh

        # Update Weights ================

        wh -= lr * dcost_wh
        bh -= lr * dcost_bh.sum(axis=0)

        wo -= lr * dcost_wo
        bo -= lr * dcost_bo.sum(axis=0)

        if epoch % 2000 == 0:
            ##### validation
            # Phase 1
            zh_t = np.dot(test_data_set, wh) + bh
            ah_t = sigmoid(zh_t)
            # Phase 2
            zo_t = np.dot(ah_t, wo) + bo
            ao_t = softmax(zo_t)
            loss = np.sum((-test_label_set) * np.log(ao_t))
            print('Loss function value: ', loss)
            error_cost.append(loss)

            if(loss<min_loss):
                min_loss=loss
                early_stop=0
            else:
                early_stop+=1
                if(early_stop>patience):
                    break
            
            

    print("train performance")
    train_performance = np.argmax(ao, axis=1)==np.argmax(train_label_set, axis=1)
    print(np.sum(train_performance)/train_performance.shape[0])

    print("test performance")
    test_performance = np.argmax(ao_t, axis=1)==np.argmax(test_label_set, axis=1)
    plt.clf()
    hist, xbins, ybins, _  = plt.hist2d(np.argmax(ao_t, axis=1), np.argmax(test_label_set, axis=1), 
        bins=len(class_names), range=((-0.5, len(class_names)-0.5), (-0.5, len(class_names)-0.5)))

    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            plt.text(xbins[j]+0.5,ybins[i]+0.5, int(hist.T[i,j]), 
                    color="w", ha="center", va="center", fontweight="bold")

    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("prediction")
    plt.ylabel("truth")
    plt.savefig("confusion-{}-{}-mfccs-life.png".format(hidden_nodes, len(class_names)))
    print(np.sum(test_performance)/test_performance.shape[0])

    

    np.savez("model-mfccs-no-res/life-{}-{}.npz".format(hidden_nodes, len(class_names)), wo=wo, bo=bo, wh=wh, bh=bh)

def main():
    #train_data_set, train_label_set, test_data_set, test_label_set = load_data(collect.top_n_freq*collect.n_chunks_per_block)
    train_data_set, train_label_set, test_data_set, test_label_set = load_data(process.n_mfcc*process.mfcc_len)
    
    train(7, train_data_set, train_label_set, test_data_set, test_label_set)

if __name__ == "__main__":
    main()
