import numpy as np
import keras,keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,GlobalAveragePooling1D
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D,Lambda
from keras import regularizers
from keras.models import load_model
import data,files,single,ens

def ens_train(in_path,out_path,n_epochs=500):
    ens.transform_template(make_model,in_path,out_path,n_epochs)

def ens_extract(seq_path,model_path,out_path):
    ens.extract_template(extract,seq_path,model_path,out_path)

def make_model(in_path,out_path,n_epochs=5):
    seq_dict=single.read_frame_feats(in_path)
    train,test=data.split_dict(seq_dict)
    X,y=get_data(train)
    params={'n_cats':y.shape[1],'ts_len':X.shape[1],'n_feats':X.shape[2]}
    model=clf_model(params)
    model.fit(X,y,epochs=n_epochs,batch_size=64)
    model.save(out_path)

def extract(seq_path,model_path,out_path):
    seq_dict=single.read_frame_feats(seq_path)
    model=load_model(model_path)
    extractor=Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)
    feat_dict={name_i:extractor.predict(np.expand_dims(seq_i,axis=0)) 
                    for name_i,seq_i in seq_dict.items()}
    single.save_ts_feats(feat_dict,out_path)

def get_data(data_dict):
    X,y=[],[]
    for name_i,seq_k in data_dict.items():
        X.append(seq_k)
        y.append( int(name_i.split('_')[0])-1)
    X=np.array(X)
    y=keras.utils.to_categorical(y)
    return X,y

def clf_model(params):
    x,input_img=basic_model(params)
    x=Dropout(0.5)(x)
    x=Dense(units=params['n_cats'],activation='softmax')(x)
    model = Model(input_img, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    model.summary()
    return model

def reg_model(params,n_units=4):
    x,input_img=basic_model(params)
#    x=Dropout(0.5)(x)
    x=Dense(units=n_units,activation='relu')(x)
    model = Model(input_img, x)
    model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(lr=0.00001))
    model.summary()
    return model

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
    return x,input_img