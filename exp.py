import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os.path 
import files,stats,spline,agum
import basic,basic.ts,ts,frames.ae

def stats_feats(in_path):
    paths=prepare_dirs(in_path,"stats",["feats"])
    stats.ens_stats(in_path,paths["feats"])

def basic_feats(in_path,n_epochs=1000):
    paths=prepare_dirs(in_path,"basic",["spline","models","feats"])
    spline.ens_upsample(in_path,paths["spline"])
    basic.ts.ens_train(paths["spline"],paths["models"],n_epochs)
    basic.ts.ens_extract(paths["spline"],paths["models"],paths["feats"])

def agum_feats(in_path,n_epochs=1000):
    paths=prepare_dirs(in_path,"agum",["spline","agum_seqs","models","feats"])
    spline.ens_upsample(in_path,paths["spline"])
    files.make_dir(paths["agum_seqs"])
    agum.ens_agum(paths["spline"],paths["agum_seqs"])
    basic.ts.ens_train(paths["agum_seqs"],paths["models"],n_epochs)
    basic.ts.ens_extract(paths["spline"],paths["models"],paths["feats"])

def sim_feats(in_path,n_epochs=350):
    paths=prepare_dirs(in_path,"sim",["spline","models","feats"])
    spline.ens_upsample(in_path,paths["spline"])
    ts.ens_train(paths["spline"],paths["models"],n_epochs)
    ts.ens_extract(paths["spline"],paths["models"],paths["feats"])

def ae_seqs(in_path,n_epochs=1000):
    paths=prepare_dirs(in_path,None,["ae","seqs"])
    frames.ae.make_model(in_path,paths["ae"],n_epochs)
    frames.ae.extract(in_path,paths["ae"],paths["seqs"])

def sim_seqs(in_path,n_epochs=350):
    paths=prepare_dirs(in_path,None,["frame_models","seqs"])
#    frames.ens_train(in_path,paths["frame_models"],n_epochs)
    frames.ens_extract(in_path,paths["frame_models"],paths["seqs"])

def basic_seqs(in_path,n_epochs=1000):
    paths=prepare_dirs(in_path,None,["frame_models","seqs"])
    basic.ens_train(in_path,paths["frame_models"],n_epochs)
    basic.ens_extract(in_path,paths["frame_models"],paths["seqs"])

def full_seqs(in_path,n_epochs=1000):
    paths=prepare_dirs(in_path,None,["frame_model","seqs"])
    basic.full_train(in_path,paths["frame_model"],n_epochs)
    files.make_dir(path["seqs"])
    basic.extract(model_path,seq_path,in_path)

def prepare_dirs(in_path,name,sufixes):
    dir_path=os.path.dirname(in_path)
    if(name):
        dir_path="%s/%s" %(dir_path,name)
        files.make_dir(dir_path)
    return get_paths(dir_path,sufixes)  

def get_paths(dir_path,sufixes):
    return {sufix_i:"%s/%s"%(dir_path,sufix_i) for sufix_i in sufixes }

if __name__ == "__main__":
#    ae_seqs("../clean/exp2/frames",n_epochs=300)
#    sim_seqs("../clean/exp2/frames",350)
    in_path="../clean/exp2/ens/seqs"
    stats_feats(in_path)
    basic_feats(in_path,1000)
    sim_feats(in_path,350)