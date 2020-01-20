import numpy as np
import keras
#import files,data,single,tools
import single,data,norm
import gen,ts.models

def make_model(in_path,out_path=None,n_epochs=5):
    (X_train,y_train),test,params=load_data(in_path,split=True)
#    X_train=np.squeeze(X_train)
    X,y=gen.full_data(X_train,y_train)
    sim_metric,model=ts.models.make_model(params)
    sim_metric.fit(X,y,epochs=n_epochs,batch_size=100)
    if(out_path):
        model.save(out_path)

def load_data(in_path,split=True):   
    feat_dict=single.read_frame_feats(in_path)
#    raise Exception(list(feat_dict.values())[0].shape)
    if(split):
        train,test=data.split(feat_dict.keys())
        train,test=prepare_data(train,feat_dict),prepare_data(test,feat_dict)
#        raise Exception(train[0].shape)
        params={'ts_len':train[0].shape[1],'n_feats':train[0].shape[2],'n_cats': train[1].shape[1]}
        return train,test,params
    else:
        names=list(feat_dict.keys())
        return prepare_data(names,feat_dict),names

def prepare_data(names,feat_dict):
    X=np.array([feat_dict[name_i] for name_i in names])
    X=norm.normalize(X,'all')
#    X=np.expand_dims(X,axis=-1)
    y=[data.parse_name(name_i)[0]-1 for name_i in names]
    y=keras.utils.to_categorical(y)
    return np.array(X),y