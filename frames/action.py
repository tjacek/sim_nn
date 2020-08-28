import keras.utils
from keras.models import load_model
from keras.models import Model
import numpy as np
import data,imgs,single
import frames,sim,gen,ens

def ens_train(in_path,out_path,n_epochs=5):
    ens.train_template(make_model,in_path,out_path,n_epochs)

def ens_extract(frame_path,model_path,out_path):
    ens.transform_template(extract,model_path,out_path,frame_path,False)

def extract(model_path,out_path,frame_path):	
    action_frames=imgs.read_frames(frame_path,True)
    model=load_model(model_path)
    extractor=Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)
    X=action_format(action_frames)
    X_feats=model.predict(X)
    names=list(action_frames.keys())
    feat_dict={ names[i]:feat_i for i,feat_i in enumerate(X_feats)}
    single.save_ts_feats(feat_dict,out_path)

def make_model(in_path,out_path,n_epochs=5,i=None):
    action_frames=imgs.read_frames(in_path,True)

    train,test=data.split_dict(action_frames)
    assert (equal_dims(train))
    X=action_format(train)

    y=np.array([ int(name_i.split("_")[0])-1 
            for name_i in list(train.keys())])

    dims=X.shape
    params={"input_shape":(dims[1],dims[2],1)} 
    if(not (i is None)):
        pair_X,pair_y=gen.binary_data(X,y,binary_cat=i,n_samples=None)
    else:
        y=keras.utils.to_categorical(y)
        pair_X,pair_y=gen.full_data(X,y)
    sim_metric,model=sim.build_siamese(params,frames.make_five)
    sim_metric.fit(pair_X,pair_y,epochs=n_epochs,batch_size=64)
    if(out_path):
        model.save(out_path)

def action_format(train):
    frames=list(train.values())
    X=np.array(frames)
    X=np.expand_dims(X, -1)
    return X

def equal_dims(train):
    dims=[ img_i.shape for img_i in list(train.values())]
    return all([dims[0]==dim_i  for dim_i in dims] )