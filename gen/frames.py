import random

def from_pairs(X_pairs,y_pairs,n_frames):
    X,y=[],[]
    for i,y_i in enumerate(y_pairs):
        pair_i=X_pairs[i]
        X_i,y_i=sample_frames(pair_i[0],pair_i[1],y_i,n_frames)
        X+=X_i
        y+=y_i
    return X,y

def sample_frames(seq_a,seq_b,y_i,n_frames):
    X,y=[],[]
    for k in range(n_frames):
        a=random.randint(0,len(seq_a)-1)
        b=random.randint(0,len(seq_b)-1)
        X.append((seq_a[a],seq_b[b]))
        y.append(y_i)
    return X,y