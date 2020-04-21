import numpy as np
import re
import imgs

def make_dataset(in_path,frames=True,full=False):
    img_seqs=imgs.read_seqs(in_path)
    format= to_dataset if(frames) else to_seq_dataset
    if(full):
        return format(img_seqs.keys(),img_seqs)
    else:
        train,test=split(img_seqs.keys())
        return format(train,img_seqs),format(test,img_seqs)

def split_dict(action_dict,selector=None):
    train,test=split(action_dict.keys(),selector=None)
    train={ name_i:action_dict[name_i] for name_i in train}
    test={ name_i:action_dict[name_i] for name_i in test}
    return train,test

def split(names,selector=None):
    if(not selector):
        selector=lambda name_i: (parse_name(name_i)[1]%2==1)
    train,test=[],[]
    for name_i in names:
        if((parse_name(name_i)[1]%2)==1):
            train.append(name_i)
        else:
            test.append(name_i)
    return train,test    

def parse_name(action_i):
    name_i=action_i.split('/')[-1]
    digits=re.findall(r'\d+',name_i)
    return int(digits[0]),int(digits[1])

def clean_name(action_i):
    name_i=action_i.split('/')[-1]
    raw=[s_i.lstrip("0") for s_i in re.findall(r'\d+',name_i)]
    return "_".join(raw)

def get_params(X,y):
    return count_cats(y),count_channels(X) 

def count_cats(y):
    return np.unique(np.array(y)).shape[0]

def count_channels(X):
    frame_dims=X[0].shape
    return int(frame_dims[-2]/frame_dims[-1])

def format_frames(frames ,n_channels=None):
    if(not n_channels):
        n_channels=count_channels(frames)        
    return np.array([np.array(np.vsplit(frame_i,n_channels)).T
                      for frame_i in frames])

def to_seq_dataset(seq_dict):
    X,y=[],[]
    for name_i in seq_dict.keys():
        cat_i=parse_name(name_i)[0]-1
        seq_i=format_frames(seq_dict[name_i])
        X.append(seq_i)
        y.append(cat_i)
    return X,y

def to_frame_dataset(seq_dict):
    X,y=[],[]
    for name_i in seq_dict.keys():
        cat_i=parse_name(name_i)[0]-1
        seq_i=format_frames(seq_dict[name_i])
        for frame_j in seq_i:
            X.append(frame_j)
            y.append(cat_i)
    return X,y