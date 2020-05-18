import numpy as np
import single,data,ens

def ens_agum(in_path,out_path):
    ens.transform_template(block_agum,in_path,out_path)

def block_agum(in_path,out_path,k=10,t=8):
    seq_dict=single.read_frame_feats(in_path)
    train,test=data.split_dict(seq_dict)
    agum_seq={}
    for name_i,seq_i in train.items():
        agum_seq[name_i]=seq_i
        for j in range(k):
            name_j="%s_%d" %(name_i,j)
            agum_seq[name_j]=sample_blocks(seq_i,t)
    agum_seq.update(test)
    single.save_frame_feats(agum_seq,out_path)

def sample_blocks(seq_i,k):
    n_blocks=int(seq_i.shape[0]/k)
    max_j=seq_i.shape[0]-k
    indexes=np.random.randint(max_j, size=n_blocks)
    indexes=np.sort(indexes)
    blocks=[seq_i[j:j+k]  for j in indexes]
    new_seq_j=np.concatenate(blocks,axis=0)
    return new_seq_j

if __name__=="__main__":
    path_i="../ens5/basic/spline"
    ens_agum(path_i,"agum_ens")