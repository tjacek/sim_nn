import numpy as np

def binary_data(X_old,y_old,binary_cat=0,n_samples=3):
    X,y=[],[]
    for i,y_i in enumerate(y_old):
        if(y_i==binary_cat):
            x_i=X_old[i]
            for j,x_j in enumerate(X_old):
                X_ij,y_ij=make_pairs(x_i,x_j,y_i,y_old[j],n_samples)
                X+=X_ij
                y+=y_ij
    X,y=np.array(X),np.array(y)
    X=[X[:,0],X[:,1]]
    return X,y

def make_pairs(x_i,x_j,y_i,y_j,n_samples=3):
    sample_i,sample_j=get_sample(x_i),get_sample(x_j)
    frames_i,frames_j= sample_i(n_samples),sample_j(n_samples)   
    X,y=[],[]
    for k,t in zip(frames_i,frames_j):
        X.append((x_i[k],x_j[t]))
        y.append(y_i==y_j)
    return X,y

def get_sample(seq_i):
    size=seq_i.shape[0]
    dist_i=get_dist(size)
    def sample_helper(n):   
        return np.random.choice(np.arange(size),n,p=dist_i)
    return sample_helper

def get_dist(n):
    inc,dec=np.arange(n),np.flip(np.arange(n))
    dist=np.amin(np.array([inc,dec]),axis=0)
    dist=dist.astype(float)
    dist=dist**2
    if(np.sum(dist)==0):
        dist.fill(1.0)
    dist/=np.sum(dist)
    return dist

def full_data(X_old,y_old):#ts
    def full_helper(i,x_i,y_i,n_samples):
        for j in range(i,n_samples):
            x_j,y_j=X_old[j],y_old[j]
            y_k=int(np.dot(y_i,y_j))                
            yield (x_i,x_j),y_k
    return template(X_old,y_old,full_helper)

def template(X_old,y_old,fun):
    X,y=[],[]
    n_samples=len(X_old)
    for i in range(n_samples):
        x_i,y_i=X_old[i],y_old[i]
        for x_ij,y_ij in fun(i,x_i,y_i,n_samples):
            X.append(x_ij)
            y.append(y_ij)
    X,y=np.array(X),np.array(y)
    X=[X[:,0],X[:,1]]
    return X,y  