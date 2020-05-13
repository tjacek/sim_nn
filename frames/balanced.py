import random,numpy as np
import frames,gen.frames,gen.balanced

def make_model(in_path,out_path=None,n_epochs=5,n_in=3,n_out=7,n_frames=5):#3
    def gen_helper(X_old,y_old):
        X,y=[],[]
        dist=gen.balanced.BalancedDist(y_old)
        for i,y_i in enumerate(y_old):
            x_i=X_old[i]
            sampled=[ dist.in_cat(y_i) for k in range(n_in)]
            sampled+=[ dist.out_cat(y_i) for k in range(n_out)]
            X_pairs,y_pairs=gen_pairs(i,sampled,X_old,y_old)
            X_i,y_i=gen.frames.squared_from_pairs(X_pairs,y_pairs,n_frames)
            X+=X_i
            y+=y_i
        X,y=np.array(X),np.array(y)
        X=[X[:,0],X[:,1]]
        return X,y
    frames.make_sim_template(in_path,out_path,n_epochs,gen_helper)

def gen_pairs(i,sampled,X_old,y_old):
    x_i=X_old[i]
    X_pairs=[ (x_i,X_old[j]) for j in sampled]
    y_pairs=[ int(y_old[i]==y_old[j]) for j in sampled]
    return X_pairs,y_pairs