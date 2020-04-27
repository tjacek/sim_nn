import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os.path 
import files,stats,spline
import basic,basic.ts,ts,frames.ae

def stats_feats(in_path):
    dir_path=os.path.dirname(in_path)
    dir_path+="/stats"
    files.make_dir(dir_path)
    stats.ens_stats(in_path,dir_path+"/feats")

def basic_feats(in_path,n_epochs=1000):
    dir_path=os.path.dirname(in_path)
    dir_path+="/basic"
    files.make_dir(dir_path)
    spline_path= dir_path+"/spline"
    spline.ens_upsample(in_path,spline_path)
    model_path= dir_path+"/models"
    basic.ts.ens_train(spline_path,model_path,n_epochs)
    feat_path=dir_path+"/feats"
    basic.ts.ens_extract(spline_path,model_path,feat_path)

def sim_feats(in_path,n_epochs=350):
    dir_path=os.path.dirname(in_path)
    dir_path+="/sim"
    files.make_dir(dir_path)
    spline_path= dir_path+"/spline"
    spline.ens_upsample(in_path,spline_path)
    model_path= dir_path+"/models"
    ts.ens_train(spline_path,model_path,n_epochs)
    feat_path=dir_path+"/feats"
    ts.ens_extract(spline_path,model_path,feat_path)

def ae_seqs(in_path,n_epochs=1000):
    dir_path=os.path.dirname(in_path)
    ae_path=dir_path+  "/ae"
    seq_path=dir_path+ "/seqs"
    frames.ae.make_model(in_path,ae_path,n_epochs)
    frames.ae.extract(in_path,ae_path,seq_path)

def sim_seqs(in_path,n_epochs=350):
    dir_path=os.path.dirname(in_path)
    model_path=dir_path+  "/frame_models"
    seq_path=dir_path +"/seqs"
    frames.ens_train(in_path,model_path,n_epochs)
    frames.ens_extract(in_path,model_path,seq_path)

def basic_seqs(in_path,n_epochs=1000):
    dir_path=os.path.dirname(in_path)
    model_path=dir_path+  "/frame_models"
    seq_path=dir_path +"/seqs"
    basic.ens_train(in_path,model_path,n_epochs)
    basic.ens_extract(in_path,model_path,seq_path)

#basic_seqs("../ens4/tmp",1000)
#ae_seqs("../time/time",n_epochs=1000)
basic_feats("../time/seqs")#,n_epochs=350)