import frames.ae
import imgs,files,exp

def img_agum(in_path,ae_model):
    full_path=in_path+"/full"
    agum_path=in_path+"/agum"
    seq_path=in_path+"/seqs"
    full_seqs=imgs.read_seqs(full_path)
    agum_seqs=imgs.read_seqs(agum_path)
    agum_seqs={files.clean_str(name_i)+"_1":seq_i 
        for name_i,seq_i in agum_seqs.items()}
    final_seqs={**full_seqs, **agum_seqs}
    frames.ae.extract(final_seqs,ae_model,seq_path)
    exp.basic_feats(seq_path)

img_agum("../agum/simple","../agum/l1/ae") 