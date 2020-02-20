import numpy as np
import keras,keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,GlobalAveragePooling1D
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D,Lambda
from keras import regularizers
import data,files,single

def ens_train(in_path,out_path,n_epochs=500):
    files.make_dir(out_path)
    for i,in_i in enumerate(files.top_files(in_path)):
        data_dict=single.read_frame_feats(in_i)
        X,y=[],[]
        for name_i,seq_k in data_dict.items():
            name_i=files.clean_str(name_i)
            if( (int(name_i.split('_')[1])%2) ==1 ):
                X.append(seq_k)
                y.append( int(name_i.split('_')[0])-1)
        X=np.array(X)
        y=keras.utils.to_categorical(y)
        params={'n_cats':y.shape[1],'ts_len':X.shape[1],'n_feats':X.shape[2]}
        model_i=basic_model(params)
        model_i.fit(X,y,epochs=n_epochs,batch_size=64)
        out_i=out_path+'/nn'+str(i)
        model_i.save(out_i)

def basic_model(params):
    activ='relu'
    input_img=Input(shape=(params['ts_len'], params['n_feats']))
    n_kerns,kern_size,pool_size=[128,128],[8,8],[4,2]
#    x = Sequential()(x)
    x=Conv1D(n_kerns[0], kernel_size=kern_size[0],activation=activ,name='conv1')(input_img)
    x=MaxPooling1D(pool_size=pool_size[0],name='pool1')(x)
    x=Conv1D(n_kerns[1], kernel_size=kern_size[1],activation=activ,name='conv2')(x)
    x=MaxPooling1D(pool_size=pool_size[1],name='pool2')(x)
    x=Flatten()(x)
    x=Dense(64, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),)(x)
    x=Dropout(0.5)(x)
    x=Dense(units=params['n_cats'],activation='softmax')(x)
    model = Model(input_img, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    model.summary()
    return model