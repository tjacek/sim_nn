import os.path
import single,files,basic

def frame_feats(frame_path,model_path,out_path=None,fun=None):
    if(not out_path):
        dst_dir=os.path.split(model_path)[0]
        out_path=dst_dir+'/seqs'	
    files.make_dir(out_path)
    if(not fun):
        fun= single.extract_frame_feats
    for i,in_i in enumerate(files.top_files(model_path)):
        out_i="%s/nn%d"%(out_path,i)
        fun(frame_path,in_i,out_i)

def ts_feats(seqs_path,out_path=None):
    if(not out_path):
        dst_dir=os.path.split(seqs_path)[0]
        out_path=dst_dir+'/feats'	
    files.make_dir(out_path)
    for i,in_i in enumerate(files.top_files(seqs_path)):
        out_i="%s/nn%d"%(out_path,i)
        single.compute_ts_feats(in_i,out_i)

frame_path='../MSR/ens1/time'
model_path='../MSR/ens1/models'
frame_feats(frame_path,model_path,fun=basic.extract)
ts_feats('../MSR/ens1/seqs')