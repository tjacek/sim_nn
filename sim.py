import keras,keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,GlobalAveragePooling1D
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D,Lambda
from keras import regularizers

def build_siamese(params,make_model):
    input_shape=params["input_shape"]#(params['ts_len'], params['n_feats'])
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    make_model(model)

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    prediction,loss=contr_loss(encoded_l,encoded_r)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    optimizer = keras.optimizers.Adam(lr = 0.00006)
    siamese_net.compile(loss=loss,optimizer=optimizer)
    extractor=Model(inputs=model.get_input_at(0),outputs=model.get_layer("hidden").output)
    extractor.summary()
    return siamese_net,extractor


def contr_loss(encoded_l,encoded_r):
    L2_layer = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)
    return L2_layer([encoded_l, encoded_r]),contrastive_loss

def contrastive_loss(y_true, y_pred):
    margin = 50
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)