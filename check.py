import numpy as np
from keras.models import load_model
from keras.models import load_model
import files,imgs

def check_model(in_path):
    model_path= in_path+"/frame_models"
    paths=files.top_files(model_path)
    model=load_model(paths[0])
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

print(count_frames("../agum/box"))
