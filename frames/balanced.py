import random,numpy as np
import frames,gen.frames
from collections import defaultdict

def make_model(in_path,out_path=None,n_epochs=5,n_in=3,n_out=10,n_frames=10):#3
    def gen_helper(X_old,y_old):
        X,y=[],[]
        dist=BalancedDist(y_old)
        for i,y_i in enumerate(y_old):
            x_i=X_old[i]
            sampled=[ dist.in_cat(y_i) for k in range(n_in)]
            sampled+=[ dist.out_cat(y_i) for k in range(n_out)]
            X_pairs,y_pairs=gen_pairs(i,sampled,X_old,y_old)
            X_i,y_i=gen.frames.from_pairs(X_pairs,y_pairs,n_frames)
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

class BalancedDist(object):
    def __init__(self,y):
        self.by_cat=sort_by_cat(y)
        self.n_cats=len(self.by_cat)

    def in_cat(self,cat_i):
        return np.random.choice(self.by_cat[cat_i])

    def out_cat(self,cat_i):
        j=random.randint(0,self.n_cats-2)
        if(j>=cat_i):
            j+=1
        return self.in_cat(j)

def sort_by_cat(y):
    by_cat=defaultdict(lambda :[])
    for i,y_i in enumerate(y):
        by_cat[y_i].append(i)
    return by_cat