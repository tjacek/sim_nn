import os.path
import frames,frames.ae
import imgs,files,exp #,data

def img_agum(in_path,ae_model):
    final_seqs,paths=get_agum_data(in_path)
    frames.ae.extract(final_seqs,ae_model,paths["seqs"])
    exp.stats_feats(paths["seqs"])
    exp.basic_feats(paths["seqs"])
    exp.stats_feats(paths["seqs"])

def ens_agum(in_path,model_path,feats=False):
    ens_path=in_path+"/ens"
    files.make_dir(ens_path)    
    final_seqs,paths=get_agum_data(in_path)
    seq_path= ens_path+"/seqs"
    final_seqs=imgs.seq_tranform(format_seq,final_seqs)
    frames.ens_extract(final_seqs,model_path,seq_path)
    if(feats):
        exp.stats_feats(seq_path)
        exp.basic_feats(seq_path)

def retrain_ae(in_path,out_path,n_epochs=1500):
    final_seqs,paths=get_agum_data(in_path)
    retrain_path="%s/retrain" % in_path
    files.make_dir(retrain_path)
    frames.ae.make_model(final_seqs,retrain_path+"/ae",n_epochs,recon=True)

def get_agum_data(in_path):
    paths=exp.get_paths(in_path,["full","agum","seqs"])
    full_seqs=imgs.read_seqs(paths["full"])
    agum_seqs=imgs.read_seqs(paths["agum"])
    agum_seqs={files.clean_str(name_i)+"_1":seq_i 
        for name_i,seq_i in agum_seqs.items()}
    final_seqs={**full_seqs, **agum_seqs}
    return final_seqs,paths

def format_seq(frame):
    size=frame.shape[1]
    new_frame=frame[size:,:]
    return new_frame

if __name__ == "__main__":
    img_agum("../agum/gap","../agum/l1/ae") 
#ens_agum("../agum/gap","../ens5/frame_models")
#exp.basic_feats("../agum/ens/seqs")
