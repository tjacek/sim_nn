import random,numpy as np
import frames,files,gen.frames

def bag_ens(in_path,out_path,k=10,n_epochs=5):
    files.make_dir(out_path)
    for i in range(k):
        out_k="%s/%d" %(out_path,i)
        make_model(in_path,out_k,n_epochs,n_seqs=4000,n_frames=10)

def make_model(in_path,out_path=None,n_epochs=5,n_seqs=4000,n_frames=10):#3
    def gen_helper(old_X,old_y):
        x_size=len(old_y)
        X,y=[],[]
        for i in range(n_seqs):
            X_i,y_i=sample_pair(old_X,old_y,x_size,n_frames)
            X+=X_i
            y+=y_i
        X,y=np.array(X),np.array(y)
        X=[X[:,0],X[:,1]]
        return X,y
    frames.make_sim_template(in_path,out_path,n_epochs,gen_helper)

def sample_pair(X,y,x_size,n_frames):
    a=random.randint(0,x_size-1)
    b=random.randint(0,x_size-1)
    seq_a,y_a=X[a],y[a]
    seq_b,y_b=X[b],y[b]
    y_i=int(y_a==y_b)
    return gen.frames.sample_frames(seq_a,seq_b,y_i,n_frames)