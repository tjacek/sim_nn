import numpy as np
import frames
import gen.balanced,gen.frames

def make_model(in_path,out_path=None,n_epochs=5,n_seqs=10,n_frames=5):#3
    def gen_helper(X_old,y_old):
#        X,y=[],[]
        dist=gen.balanced.BalancedDist(y_old)
        sampled=sample_pairs(dist,n_seqs)
        X_pairs=[ (X_old[a],X_old[b]) for a,b in sampled]
        y_pairs=[ int(y_old[a]==y_old[b]) for a,b in sampled]
        X,y=gen.frames.squared_from_pairs(X_pairs,y_pairs,n_frames)
#        raise Exception(len(y))
        X,y=np.array(X),np.array(y)
        X=[X[:,0],X[:,1]]
        return X,y
    frames.make_sim_template(in_path,out_path,n_epochs,gen_helper)

def sample_pairs(dist,n_frames):
    pairs=[]
    for a_i,b_i in gen_pairs(dist.n_cats):
        pairs+=zip(dist.sample(n_frames,a_i),dist.sample(n_frames,b_i))
#    raise Exception( len(pairs) )
    return pairs

def gen_pairs(n_cats):
    pairs=[]
    for i in range(n_cats):
        for j in range(i,n_cats):
            pairs.append((i,j))
    return pairs