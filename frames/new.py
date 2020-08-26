import keras
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Add,Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D,Activation,Lambda
from keras import regularizers
import data,imgs,sim,ens,gen

class SimFrames(object):
    def __init__(self,gen_pairs,make_model):
        self.gen=gen_pairs
        self.make_model=make_model

    def __call__(self,in_path,out_path,n_epochs,cat_i):
        frames=imgs.read_seqs(in_path)
        train,test=data.split_dict(frames)
        X,y=data.to_seq_dataset(train)
        X,y=self.gen(X,y,cat_i) 
        n_channels=X[0].shape[-1]
        params={"input_shape":(X[0].shape[1],X[0].shape[2],n_channels)} 
        sim_metric,model=sim.build_siamese(params,self.make_model)
        sim_metric.fit(X,y,epochs=n_epochs,batch_size=100)
        if(out_path):
            model.save(out_path)  

    def ens_train(self,in_path,out_path,n_epochs=5):
        ens.train_template(self,in_path,out_path,n_epochs)

def get_sim_frames(n_samples=3):
    assert(type(n_samples)==int)
    def gen_helper(X,y,cat_i):
        return gen.binary_data(X,y,cat_i,n_samples)
    def model_helper(model):
        params={'kern':(3,3),'pool':(2,2),'filters':[32,16,16,16]}	
        return make_nn(model,params)
    return SimFrames(gen_helper,model_helper)

def make_nn(model,params=None):
    activ='relu'
    kern_size,pool_size,filters=params['kern'],params['pool'],params['filters']
    for filtr_i in filters:
        model.add(Conv2D(filtr_i, kern_size, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size, padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation=activ,name='hidden',kernel_regularizer=regularizers.l1(0.01)))
    return model
