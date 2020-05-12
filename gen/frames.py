import random
import gen

def squared_from_pairs(X_pairs,y_pairs,n_frames):
    return from_pairs_template(X_pairs,y_pairs,n_frames,squared_pairs)

def ordered_from_pairs(X_pairs,y_pairs,n_frames):
    return from_pairs_template(X_pairs,y_pairs,n_frames,ordered_frames)

def from_pairs(X_pairs,y_pairs,n_frames):
    return from_pairs_template(X_pairs,y_pairs,n_frames,sample_frames)

def from_pairs_template(X_pairs,y_pairs,n_frames,fun):
    X,y=[],[]
    for i,y_i in enumerate(y_pairs):
        pair_i=X_pairs[i]
        X_i,y_i=fun(pair_i[0],pair_i[1],y_i,n_frames)
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

def ordered_frames(seq_a,seq_b,y_i,n_frames):
    a=[ random.randint(0,len(seq_a)-1) for k in range(n_frames)]
    b=[ random.randint(0,len(seq_b)-1) for k in range(n_frames)]
    a.sort()
    b.sort()
    X,y=[],[]
    for a_j,b_j in zip(a,b):
        X.append((seq_a[a_j],seq_b[b_j]))
        y.append(y_i)
    return X,y

def squared_pairs(seq_a,seq_b,y_i,n_frames):
#    raise Exception(n_frames)	
    sample_i,sample_j=gen.get_sample(seq_a),gen.get_sample(seq_b)
    frames_i,frames_j= sample_i(n_frames),sample_j(n_frames)   
    X,y=[],[]
    for k,t in zip(frames_i,frames_j):
        X.append((seq_a[k],seq_b[t]))
        y.append(y_i)
    return X,y