import keras.utils
import numpy as np
import data,imgs
import frames,sim,gen

def action_img(in_path,out_path,n_epochs=5):
    action_frames=imgs.read_frames(in_path,True)
    train,test=data.split_dict(action_frames)
    X=data.format_frames(list(train.values())) 
    y=[ int(name_i.split("_")[0])-1 
            for name_i in list(train.keys())]
    y=keras.utils.to_categorical(y)
    params={"input_shape":(64,64,X[0].shape[-1])} 
    pair_X,pair_y=gen.full_data(X,y)
    sim_metric,model=sim.build_siamese(params,frames.make_five)
    sim_metric.fit(pair_X,pair_y,epochs=n_epochs,batch_size=64)
    if(out_path):
        model.save(out_path)