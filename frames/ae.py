import numpy as np
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input, Dense,Conv2D,Reshape,Conv2DTranspose
from keras.layers import Flatten,MaxPooling2D,UpSampling2D
from keras.models import load_model
import imgs,data

def make_model(in_path,out_path=None,n_epochs=1000,recon=True):
    frames=imgs.read_seqs(in_path)
    train,test=data.split_dict(frames)
    X,y=data.to_frame_dataset(train)
    X=np.array(X)
    params={'n_channels':X.shape[-1]}
    model,recon=make_autoencoder(params)
    model.summary()
    model.fit(X,X,epochs=n_epochs,batch_size=256)
    model.save(out_path)
    if(recon):
        recon.save(out_path+"_recon")

def make_autoencoder(params):
    input_img = Input(shape=(64, 64, params['n_channels']))
    n_kerns=32
    x = Conv2D(n_kerns, (5, 5), activation='relu',padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    shape = K.int_shape(x)
    x=Flatten()(x)
    encoded=Dense(100)(x)    
    x = Dense(shape[1]*shape[2]*shape[3])(encoded)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (5, 5), activation='relu',padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(n_kerns, (5, 5), activation='relu',padding='same')(x)
    
    x=Conv2DTranspose(filters=params['n_channels'],kernel_size=n_kerns,padding='same')(x)
    recon=Model(input_img,encoded)
    autoencoder = Model(input_img, x) 
    autoencoder.compile(optimizer='adam',#keras.optimizers.SGD(lr=0.0001,  momentum=0.9, nesterov=True), 
                      loss='mean_squared_error')
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

#def extract_feats(in_path,model_path,out_path=None):
#    if(not out_path):
#        out_path=os.path.split(in_path)[0]+'/ae_feats'
#    model=load_model(model_path)
#    seq_dict=imgs.read_seqs(in_path) 
#    feat_dict=extract.frame_features(seq_dict,model)
#    extract.save_seqs(feat_dict,out_path)

#def add_noise(X):
#    std=0.25*np.mean(X)
#    noise = np.random.normal(loc=0.0, scale=std, size=X.shape)
#    return X+noise