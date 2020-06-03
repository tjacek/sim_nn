from keras.layers import Input,Dense,Reshape,Lambda
from keras.layers import Conv2D,Conv2DTranspose
from keras.layers import Flatten,MaxPooling2D,UpSampling2D
from keras.losses import mse
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
    model=make_autoencoder(params)

    original_dim = 64 * 64

    model.summary()
    model.fit(X,epochs=n_epochs,batch_size=64)
    model.save(out_path)

def make_autoencoder(params):
    input_img = (64, 64, params['n_channels'])
    inputs = Input(shape=input_img, name='encoder_input')
    n_kerns=32
    x = Conv2D(32, (5, 5), activation='relu',padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    shape = K.int_shape(x)

    x=Flatten()(x)

    z_mean = Dense(100, name='z_mean')(x)
    z_log_var = Dense(100, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(100,), name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean], name='encoder')
    encoder.summary()
    latent_inputs = Input(shape=(100,), name='z_sampling')
    x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (5, 5), activation='relu',padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (5, 5), activation='relu',padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, (5, 5), activation='relu',padding='same')(x)

    outputs=Conv2DTranspose(filters=params['n_channels'],kernel_size=n_kerns,padding='same')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    outputs = decoder(encoder(inputs))
    vae = Model(inputs, outputs, name='vae_mlp')
    vae.summary()

    reconstruction_loss = mse(inputs, outputs)
#    reconstruction_loss *= (64*64)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss) + K.mean(kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae