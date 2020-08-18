import numpy as np
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input, Dense,Conv2D,Reshape,Conv2DTranspose
from keras.layers import Flatten,MaxPooling2D,UpSampling2D
from keras.models import load_model
import imgs,data,single
from keras import regularizers

def make_model(frames,out_path=None,n_epochs=1000,recon=True):
    if(type(frames)==str):
        frames=imgs.read_seqs(frames)
    train,test=data.split_dict(frames)
    X,y=data.to_frame_dataset(train)
    X=sub_sample(X)
    X=np.array(X)
#    add_noise(X)
    params={'n_channels':X.shape[-1],"dim":(X.shape[1],X.shape[2])}
    model,auto=make_autoencoder(params)
    model.summary()
    model.fit(X,X,epochs=n_epochs,batch_size=16)
    auto.save(out_path)
    if(recon):
        model.save(out_path+"_recon")

def sub_sample(X):
    return [ x_i for i,x_i in enumerate(X)
                if( (i%3)==0)] 

def make_autoencoder(params):
#    raise Exception(params)
    x,y=params["dim"]
    input_img = Input(shape=(x,y, params['n_channels']))
    n_kerns=32
    x = Conv2D(n_kerns, (5, 5), activation='relu',padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    shape = K.int_shape(x)
    x=Flatten()(x)
    encoded=Dense(100,name='hidden',kernel_regularizer=regularizers.l1(0.01))(x)    
    x = Dense(shape[1]*shape[2]*shape[3])(encoded)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (5, 5), activation='relu',padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(n_kerns, (5, 5), activation='relu',padding='same')(x)
    
    x=Conv2DTranspose(filters=params['n_channels'],kernel_size=n_kerns,padding='same')(x)
    recon=Model(input_img,encoded)
    autoencoder = Model(input_img, x)

    autoencoder.compile(optimizer='adam',
                      loss='mean_squared_error')#CustomLoss(autoencoder)
    return autoencoder,recon

def reconstruct(in_path,model_path,out_path=None,diff=False):
    frames=imgs.read_seqs(in_path)
    model=load_model(model_path)
    frames={ name_i:data.format_frames(seq_i)
                for name_i,seq_i in frames.items()}
    rec_frames={}
    for name_i,seq_i in frames.items():
        rec_seq_i=model.predict(seq_i)
        rec_seq_i=  [np.vstack(frame_j.T) 
                        for frame_j in rec_seq_i]
        rec_frames[name_i]=rec_seq_i
    imgs.save_seqs(rec_frames,out_path)

def extract(seq_dict,model_path,out_path=None):
    model=load_model(model_path)
    if(type(seq_dict)==str):
        seq_dict=imgs.read_seqs(seq_dict)
    feat_dict=single.extractor_template(seq_dict,model)
    single.save_frame_feats(feat_dict,out_path)

def add_noise(X):
    std=0.25*np.mean(X)
    noise = np.random.normal(loc=0.0, scale=std, size=X.shape)
    return X+noise

#class CustomLoss(object):
#    def __init__(self,model):
#        self.model=model
#
#    def __call__(self,y_pred, y_true, sample_weight=None):
#        mse = K.mean(K.square(y_true - y_pred), axis=1)
#        lam = 1e-4
#        W = K.variable(value=self.model.get_layer('hidden').get_weights()[0])  # N x N_hidden
#        W = K.transpose(W)  # N_hidden x N
#        h = self.model.get_layer('hidden').output
#        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
#        contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)
#        raise Exception(y_pred.shape)
#        return mse #+ contractive