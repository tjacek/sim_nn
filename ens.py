import numpy as np
from keras.models import load_model
import imgs,data,files

def extract_feats(frame_path,model_path,out_path=None):
    extractor=load_model(model_path)
    img_seqs=imgs.read_seqs(frame_path)
    feats_seq={name_i:data.format_frames(seq_i)  
                for name_i,seq_i in img_seqs.items()}
    feat_dict={name_i:extractor.predict(seq_i) 
                for name_i,seq_i in feats_seq.items()}
    save_seqs(feat_dict,out_path)

def save_seqs(feat_dict,out_path):
    files.make_dir(out_path)
    for name_j,seq_j in feat_dict.items():
        name_j=name_j.split('.')[0]
        out_j=out_path+'/'+name_j
        np.save(out_j,seq_j)

frame_path='../../sim3/time'
model_path='../../sim3/ens3/models/nn0'
extract_feats(frame_path,model_path,'test')