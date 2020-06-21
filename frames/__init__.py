import keras
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Add,Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D,Activation,Lambda
from keras import regularizers
import imgs,data,gen,single,ens
import sim,files#ts.models
from keras.models import load_model

def ens_train(in_path,out_path,n_epochs=5):
    ens.train_template(make_model,in_path,out_path,n_epochs)

def ens_extract(frame_path,model_path,out_path):
    ens.transform_template(extract,model_path,out_path,frame_path,False)

def extract(model_path,out_path,frames):
    if(type(frames)==str):
        frames=imgs.read_seqs(frames)
    model=load_model(model_path)
    extractor=Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)
    feat_dict=single.extractor_template(frames,extractor)
    single.save_frame_feats(feat_dict,out_path)

def make_model(in_path,out_path=None,n_epochs=5,cat_i=0,n_samples=3):#3
    def gen_helper(X,y):
        return gen.binary_data(X,y,cat_i,n_samples)
    make_sim_template(in_path,out_path,n_epochs,gen_helper)

def make_sim_template(in_path,out_path,n_epochs,gen_pairs):
    frames=imgs.read_seqs(in_path)
    train,test=data.split_dict(frames)
    X,y=data.to_seq_dataset(train)
    X,y=gen_pairs(X,y) #gen.binary_data(X,y,cat_i,n_samples)
    n_channels=X[0].shape[-1]
    params={"input_shape":(64,64,n_channels)} 
    sim_metric,model=sim.build_siamese(params,make_five)
    sim_metric.fit(X,y,epochs=n_epochs,batch_size=100)
    if(out_path):
        model.save(out_path)    

def make_five(model):
    activ='relu'
    kern_size,pool_size,filters=(3,3),(2,2),[32,16,16,16]
    for filtr_i in filters:
        model.add(Conv2D(filtr_i, kern_size, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size, padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation=activ,name='hidden',kernel_regularizer=regularizers.l1(0.01)))
    return model
