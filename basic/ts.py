import numpy as np
import keras,keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,GlobalAveragePooling1D
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D,Lambda
from keras import regularizers
from keras.models import load_model
import data,files,single

def ens_train(in_path,out_path,n_epochs=500):
    files.make_dir(out_path)
    for i,in_i in enumerate(files.top_files(in_path)):
        out_i='%s/nn%d' (out_path,i)
        make_model(in_i,out_i,n_epochs)

def make_model(in_path,out_path,n_epochs=5):
    seq_dict=single.read_frame_feats(in_path)
    train,test=data.split_dict(seq_dict)
    X,y=get_data(seq_dict)
    params={'n_cats':y.shape[1],'ts_len':X.shape[1],'n_feats':X.shape[2]}
    model=basic_model(params)
    model.fit(X,y,epochs=n_epochs,batch_size=64)
    model.save(out_path)

def ens_extract(frame_path,model_path,feat_path):
    files.make_dir(feat_path)
    for i,in_i in enumerate(files.top_files(frame_path)):
        seq_dict=single.read_frame_feats(in_i)
        (X,y),names=get_data(seq_dict),list(seq_dict.keys())
        model_i=load_model(model_path+'/nn'+str(i))
        extr_i=Model(inputs=model_i.input,
                outputs=model_i.get_layer("hidden").output)
        feats_i=extr_i.predict(X)
        feat_dict_i={ names[j]:sample_ij for j,sample_ij in enumerate(feats_i)}
        single.save_ts_feats(feat_dict_i,feat_path+'/nn'+str(i))

def get_data(data_dict):
    X,y=[],[]
    for name_i,seq_k in data_dict.items():
        X.append(seq_k)
        y.append( int(name_i.split('_')[0])-1)
    X=np.array(X)
    y=keras.utils.to_categorical(y)
    return X,y

def basic_model(params):
    activ='relu'
    input_img=Input(shape=(params['ts_len'], params['n_feats']))
    n_kerns,kern_size,pool_size=[128,128],[8,8],[4,2]
    x=Conv1D(n_kerns[0], kernel_size=kern_size[0],activation=activ,name='conv1')(input_img)
    x=MaxPooling1D(pool_size=pool_size[0],name='pool1')(x)
    x=Conv1D(n_kerns[1], kernel_size=kern_size[1],activation=activ,name='conv2')(x)
    x=MaxPooling1D(pool_size=pool_size[1],name='pool2')(x)
    x=Flatten()(x)
    x=Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),)(x)
    x=Dropout(0.5)(x)
    x=Dense(units=params['n_cats'],activation='softmax')(x)
    model = Model(input_img, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    model.summary()
    return model