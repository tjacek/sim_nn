import keras
import files,ts,ts.models,ens

def make_feats(in_path,out_path,n_epochs=350,n_cats=20):
    files.make_dir(out_path)
    spline_path= out_path+'/spline'
    spline.ens_upsample(in_path,spline_path)
    model_path=out_path+'/models'
    fun=ts.make_binary_model
    ens.train_template(fun,spline_path,out_path,n_epochs,n_cats)

def ens_extract(frame_path,model_path,out_path=None):  
    ens.transform_template(extract,model_path,out_path,frame_path,dir_ensemble=False)

def extract(model_path,out_path,frame_path):
    ts.extract(frame_path,model_path,out_path)

def make_model(in_path,out_path=None,n_epochs=5,cat_i=None):
    (X_train,y_train),test,params=ts.load_data(in_path,split=True)
    y=y_train[:,cat_i]
    y=keras.utils.to_categorical(y)
    X,y=gen.full_data(X_train,y)
    sim_metric,model=ts.models.make_model(params)
    sim_metric.fit(X,y,epochs=n_epochs,batch_size=64)
    if(out_path):
        model.save(out_path)