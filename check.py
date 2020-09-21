import numpy as np
import os.path
from keras.models import load_model
from collections import defaultdict
import files,imgs,data

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

check_model("../fourth_exp/scale/ae")
