import keras
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Add,Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D,Activation,Lambda
from keras import regularizers
import imgs,data,gen,single
import ts.models
from keras.models import load_model

def extract(frame_path,model_path,out_path):
    frames=imgs.read_frames(frame_path,as_dict=True)
    X=data.format_frames(list(frames.values()))
    names=list(frames.keys())
    extractor=load_model(model_path)
    X_feats=extractor.predict(X)
    feat_dict={ names[i]:feat_i for i,feat_i in enumerate(X_feats)}
    single.save_ts_feats(feat_dict,out_path)

def make_model(in_path,out_path=None,n_epochs=5):
    frames=imgs.read_frames(in_path,as_dict=True)
    train,test=data.split_dict(frames)
    X=data.format_frames(list(train.values()))
    y=[ int(name_i.split("_")[0])-1 for name_i in train.keys()]
    X,y=gen.full_data(X,y)
    n_channels=X[0].shape[-1]
    sim_metric,model=make_five(20,n_channels,params=None)
    sim_metric.fit(X,y,epochs=n_epochs,batch_size=100)
    if(out_path):
        model.save(out_path)

def make_five(n_cats,n_channels,params=None):
    if(not params):
        params={}
    input_shape=(64,64,n_channels)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    activ='relu'
    kern_size,pool_size,filters=(3,3),(2,2),[32,16,16,16]
    for filtr_i in filters:
        model.add(Conv2D(filtr_i, kern_size, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size, padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation=activ,name='hidden',kernel_regularizer=regularizers.l1(0.01)))

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    prediction,loss=ts.models.contr_loss(encoded_l,encoded_r)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    optimizer = keras.optimizers.Adam(lr = 0.00006)#keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True)
    siamese_net.compile(loss=loss,#"binary_crossentropy",
        optimizer=optimizer)
    extractor=Model(inputs=model.get_input_at(0),outputs=model.get_layer("hidden").output)
    extractor.summary()
    return siamese_net,extractor