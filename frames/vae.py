from keras.layers import Input, Dense,Conv2D,Reshape,Conv2DTranspose,Lambda
from keras.models import Model
from keras import backend as K
import numpy as np
import imgs,data

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def make_model(in_path,out_path=None,n_epochs=1000,recon=True):
    frames=imgs.read_seqs(in_path)
    train,test=data.split_dict(frames)
    X,y=data.to_frame_dataset(train)
    X=np.array(X)
#    add_noise(X)
    params={'n_channels':X.shape[-1]}
    model,auto=make_autoencoder(params)
    model.summary()
    model.fit(X,X,epochs=n_epochs,batch_size=64)
    auto.save(out_path)
    if(recon):
        model.save(out_path+"_recon")

def make_autoencoder(params):
    input_img = (64, 64)#, params['n_channels'])
    intermediate_dim = 512
    latent_dim = 64
    original_dim = 64 * 64
    inputs = Input(shape=input_img, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
#    encoder.summary()
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    
    vae.summary()
    raise Exception("OK")