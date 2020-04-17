import numpy as np
from keras.models import load_model
import scipy.stats
import imgs,data,files

def compute_ts_feats(in_path,out_path):
    data_dict=read_frame_feats(in_path)
    feat_dict={ name_i:feat_vector(seq_i)
                    for name_i,seq_i in data_dict.items()}
    save_ts_feats(feat_dict,out_path)

def feat_vector(seq_j):
    feats=[]
    for ts_k in seq_j.T:
    	feats+=EBTF(ts_k)
    return np.array(feats)

def EBTF(feat_i):
    if(np.all(feat_i==0)):
        return [0.0,0.0,0.0,0.0]
    return [np.mean(feat_i),np.std(feat_i),
    	                scipy.stats.skew(feat_i),time_corl(feat_i)]

def time_corl(feat_i):
    n_size=feat_i.shape[0]
    x_i=np.arange(float(n_size),step=1.0)
    return scipy.stats.pearsonr(x_i,feat_i)[0]

def read_frame_feats(in_path):
    seq_dict={}
    for path_i in files.top_files(in_path):
        postfix=in_path.split(".")[-1]
        if(postfix=="npy"):
            seq_i=np.loadtxt(path_i,delimiter=',')
        else:
            seq_i=np.load(path_i)
        name_i=files.clean_str(path_i.split('/')[-1])
        seq_dict[name_i]=seq_i
    return seq_dict

def save_ts_feats(feat_dict,out_path):
    lines=[]
    for name_j,feat_j in feat_dict.items():
        line_i=np.array2string(feat_j,separator=",")
        line_i=line_i.replace('\n',"")+'#'+name_j
        lines.append(line_i)
    feat_txt='\n'.join(lines)
    feat_txt=feat_txt.replace('[','').replace(']','')
    file_str = open(out_path,'w')
    file_str.write(feat_txt)
    file_str.close()

def extract_frame_feats(frame_path,model_path,out_path=None):
    extractor=load_model(model_path)
    img_seqs=imgs.read_seqs(frame_path)
    feat_dict=extractor_template(img_seqs,extractor)
    save_frame_feats(feat_dict,out_path)

def extractor_template(img_seqs,extractor):
    feats_seq={name_i:data.format_frames(seq_i)  
                for name_i,seq_i in img_seqs.items()}
    feat_dict={name_i:extractor.predict(seq_i) 
                for name_i,seq_i in feats_seq.items()}
    return feat_dict

def save_frame_feats(feat_dict,out_path):
    files.make_dir(out_path)
    for name_j,seq_j in feat_dict.items():
        name_j=name_j.split('.')[0]
        out_j=out_path+'/'+name_j
        np.save(out_j,seq_j)

if __name__ == "__main__":
    compute_ts_feats('test/seqs','test/basic.txt')