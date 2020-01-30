import numpy as np
import keras
import imgs,data,single
from keras.models import Model,Sequential
from keras.layers import Input,Add,Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D, Activation
from keras import regularizers
from keras.models import load_model

def extract(frame_path,model_path,out_path):
    seq_dict=imgs.read_seqs(frame_path)
    model=load_model(model_path)
    extractor=Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)
    frame_feat_dict={}
    for name_i,seq_i in seq_dict.items():
        seq_i=data.format_frames(seq_i)
        frame_feat_dict[name_i]=extractor.predict(seq_i)
    single.save_frame_feats(frame_feat_dict,out_path)

def train_model(in_path,out_path,n_epochs=5):
    (X_train,y_train),test=data.make_dataset(in_path,frames=True,full=False)
    X_train=data.format_frames(X_train)
    y_train=keras.utils.to_categorical(y_train)   
    n_cats,n_channels=y_train.shape[1],X_train.shape[-1]
    model=make_model(n_cats,n_channels)#,params=None)
    model.fit(X_train,y_train,epochs=n_epochs,batch_size=100)
    if(out_path):
        model.save(out_path)

def make_model(n_cats,n_channels): #,params=None):
    input_img = Input(shape=(64, 64, n_channels))
    x=input_img
    kern_size,pool_size,filters=(3,3),(2,2),[32,16,16,16]
    for filtr_i in filters:
        x = Conv2D(filtr_i, kern_size, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size, padding='same')(x)
    x=Flatten()(x)
    x=Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),)(x)
    x=Dropout(0.5)(x)
    x=Dense(units=n_cats,activation='softmax')(x)
    model = Model(input_img, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    model.summary()
    return model

def sample_seq(frames,size=5):
    n_frames=len(frames)
    dist=get_dist(n_frames)
    def sample(n):   
        return np.random.choice(np.arange(n_frames),n,p=dist)
    return sample(size)

def get_dist(n):
    inc,dec=np.arange(n),np.flip(np.arange(n))
    dist=np.amin(np.array([inc,dec]),axis=0)
    dist=dist.astype(float)
    dist=dist**2
    dist/=np.sum(dist)
    return dist