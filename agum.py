import os.path
import frames,frames.ae
import imgs,files,exp #,data

def simple_agum(in_path,ae_model):
    paths={path_i:"%s/%s" % (in_path,path_i) 
            for path_i in ["frames","seqs"]}
    frames.ae.extract(paths["frames"],ae_model,paths["seqs"])
    exp.basic_feats(paths["seqs"])
    exp.stats_feats(paths["seqs"])

def unify_agum(paths,ae_model,out_path):
    img_dict=[ imgs.read_seqs(path_i)
                for path_i in paths]
    img_dict=[files.clean_dict(dict_i) 
                for dict_i in img_dict]
    agum_set=img_dict[0]
    for i,dict_i in enumerate(img_dict[1:]):
        for name_j,seq_j in dict_i.items():
            if(in_train(name_j)):
                name_j="%s_%d" % (name_j,i)
                agum_set[name_j]=seq_j
    files.make_dir(out_path)
    seq_path="%s/%s" % (out_path,"frames")
    imgs.save_seqs(agum_set,seq_path)
    simple_agum(out_path,ae_model)

def in_train(name_j):
#    name_j=files.clean_str(name_j)
    person_j=int(name_j.split("_")[1])
    return (person_j %2)==1
#def img_agum(in_path,ae_model):
#    final_seqs,paths=get_agum_data(in_path)
#    frames.ae.extract(final_seqs,ae_model,paths["seqs"])
#    exp.stats_feats(paths["seqs"])
#    exp.basic_feats(paths["seqs"])
#    exp.stats_feats(paths["seqs"])

#def ens_agum(in_path,model_path,feats=False):
#    ens_path=in_path+"/ens"
#    files.make_dir(ens_path)    
#    final_seqs,paths=get_agum_data(in_path)
#    seq_path= ens_path+"/seqs"
#    final_seqs=imgs.seq_tranform(format_seq,final_seqs)
#    frames.ens_extract(final_seqs,model_path,seq_path)
#    if(feats):
#        exp.stats_feats(seq_path)
#        exp.basic_feats(seq_path)

#def retrain_ae(in_path,out_path,n_epochs=1500):
#    final_seqs,paths=get_agum_data(in_path)
#    retrain_path="%s/retrain" % in_path
#    files.make_dir(retrain_path)
#    frames.ae.make_model(final_seqs,retrain_path+"/ae",n_epochs,recon=True)

#def get_agum_data(in_path):
#    paths=exp.get_paths(in_path,["full","agum","seqs"])
#    full_seqs=imgs.read_seqs(paths["full"])
#    agum_seqs=imgs.read_seqs(paths["agum"])
#    agum_seqs={files.clean_str(name_i)+"_1":seq_i 
#        for name_i,seq_i in agum_seqs.items()}
#    final_seqs={**full_seqs, **agum_seqs}
#    return final_seqs,paths

#def format_seq(frame):
#    size=frame.shape[1]
#    new_frame=frame[size:,:]
#    return new_frame

if __name__ == "__main__":
    ae_model="../common/ae"
#    simple_agum("../short",ae_model)
    paths=["../full","../short/frames"]
    unify_agum(paths,ae_model,"test")