import numpy as np
import scipy.stats
import ens,single

def ens_stats(in_path,out_path):
    ens.transform_template(compute_ts_feats,in_path,out_path)

def compute_ts_feats(in_path,out_path):
    data_dict=single.read_frame_feats(in_path)
    feat_dict={ name_i:feat_vector(seq_i)
                    for name_i,seq_i in data_dict.items()
                            if(seq_i.shape[0]>1)}
    single.save_ts_feats(feat_dict,out_path)

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

if __name__ == "__main__":
    ens_stats('../ens/seqs','../ens/stats')