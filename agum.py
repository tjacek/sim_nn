import os.path
import frames,frames.ae
import imgs,files,exp

def img_agum(in_path,ae_model):
    final_seqs,paths=get_agum_data(in_path)
    frames.ae.extract(final_seqs,ae_model,paths["seqs"])
    exp.basic_feats(paths["seqs"])

def ens_agum(in_path,model_path):
    dir_path=os.path.split(in_path)[0]
    ens_path=dir_path+"/ens"
    files.make_dir(ens_path)    
    final_seqs,paths=get_agum_data(in_path)
    seq_path= ens_path+"/seqs"
    frames.ens_extract(final_seqs,model_path,seq_path)

def get_agum_data(in_path):
    paths=exp.get_paths(in_path,["full","agum","seqs"])
    full_seqs=imgs.read_seqs(paths["full"])
    agum_seqs=imgs.read_seqs(paths["agum"])
    agum_seqs={files.clean_str(name_i)+"_1":seq_i 
        for name_i,seq_i in agum_seqs.items()}
    final_seqs={**full_seqs, **agum_seqs}
    return final_seqs,paths

#img_agum("../agum/simple","../agum/l1/ae") 
ens_agum("../agum/simple","../ens5/frame_models")