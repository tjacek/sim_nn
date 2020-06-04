import numpy as np
import keras
from keras.models import load_model
import single,data,norm,files
import gen,ts.models,ens

def ens_extract(frame_path,model_path,out_path=None):  
    ens.extract_template(extract,frame_path,model_path,out_path)

def extract(frame_path,model_path,out_path):
    (X,y),names=load_data(frame_path,split=False)
    extractor=load_model(model_path)
    X_feats=extractor.predict(X)
    feat_dict={ names[i]:feat_i for i,feat_i in enumerate(X_feats)}
    single.save_ts_feats(feat_dict,out_path)

def ens_train(in_path,out_path,n_epochs=5):
    ens.transform_template(make_model,in_path,out_path,n_epochs)

def make_model(in_path,out_path=None,n_epochs=5,cat_i=None):
    (X_train,y_train),test,params=load_data(in_path,split=True)
    X,y=gen.full_data(X_train,y_train)
    raise Exception(y.shape)
    sim_metric,model=ts.models.make_model(params)
    sim_metric.fit(X,y,epochs=n_epochs,batch_size=100)
    if(out_path):
        model.save(out_path)

def make_binary_model(in_path,out_path=None,n_epochs=5,cat_i=None):
    (X_train,y_train),test,params=load_data(in_path,split=True)
    y=y_train[:,cat_i]
    y=keras.utils.to_categorical(y)
    X,y=gen.full_data(X_train,y)
    sim_metric,model=ts.models.make_model(params)
    sim_metric.fit(X,y,epochs=n_epochs,batch_size=64)
    if(out_path):
        model.save(out_path)

def load_data(in_path,split=True):   
    feat_dict=single.read_frame_feats(in_path)
    if(split):
        train,test=data.split(feat_dict.keys())
        train,test=prepare_data(train,feat_dict),prepare_data(test,feat_dict)
        params={'ts_len':train[0].shape[1],'n_feats':train[0].shape[2],'n_cats': train[1].shape[1]}
        return train,test,params
    else:
        names=list(feat_dict.keys())
        return prepare_data(names,feat_dict),names

def prepare_data(names,feat_dict):
    X=np.array([feat_dict[name_i] for name_i in names])
    X=norm.normalize(X,'all')
    y=[data.parse_name(name_i)[0]-1 for name_i in names]
    y=keras.utils.to_categorical(y)
    return np.array(X),y