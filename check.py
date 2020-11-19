import numpy as np
import os.path
from keras.models import Model
from keras.models import load_model
from collections import defaultdict
import files,imgs,data

def check_1DCNN(in_path):
    model=load_model(in_path)
    model.summary()
    input_shape=model.input_shape
    input_shape=(1,input_shape[1],input_shape[2])
    test=np.zeros(input_shape)
    test[:,0,:]=1
    print(np.sum(test[0],axis=0))
    extractor=Model(inputs=model.input,
                outputs=model.layers[1].output)
    extractor.summary()
    x=extractor.predict(test)[0]
    print(x[:,])

def check_filters(in_path):
    model=load_model(in_path)
    model.summary()
    filters=model.layers[3].get_weights()[0]
    print(filters.shape)
    filters=np.abs(filters)
    filters=np.sum(filters,axis=0)
    print(filters.shape)
    max_feat=[ np.amax(filtr_i)/np.sum(filtr_i) 
                for filtr_i in filters.T]
    print(max_feat)

def check_model(in_path):
    if(os.path.isdir(in_path)):
        model_path= in_path+"/frame_models"
        paths=files.top_files(model_path)
        model=load_model(paths[0])
    else:
        model=load_model(in_path)
    model.summary()

def get_proportion(in_path):
    img_seq=imgs.read_seqs(in_path)
    dims=[seq_i[0].shape for seq_i in img_seq.values()]
    prop=[ dim_i[0]/dim_i[1] for dim_i in dims]
    print("mean%s median%s" % (np.mean(prop),np.median(prop)))

def count_frames(in_path):
    img_seq=imgs.read_seqs(in_path)
    seq_len=[len(seq_i) for seq_i in img_seq.values()]
    return sum(seq_len)

def compare_lenght(in_path):
    seq_dict=imgs.read_seqs(in_path)
    len_dict=get_len_dict(seq_dict)
    train,test=data.split(len_dict.keys())
    train,test=by_cat(train),by_cat(test)
    for cat_i in train.keys():
        train_i=np.mean([len_dict[name_i] for name_i in train[cat_i]])
        test_i=np.mean([len_dict[name_i] for name_i in test[cat_i]])
        print("%d,%.2f,%.2f" % (cat_i,test_i,train_i))

def by_cat(names):
    cat_dict=defaultdict(lambda:[])
    for name_i in names:
        cat_i=int(name_i.split("_")[0])-1
        cat_dict[cat_i].append(name_i)
    return cat_dict

def get_len_dict(seq_dict):
    return { files.clean_str(name_i):len(seq_i) 
                for name_i,seq_i in seq_dict.items()}

check_filters("test/binary_1D_CNN/nn0")
