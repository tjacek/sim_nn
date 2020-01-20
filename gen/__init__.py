import numpy as np

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