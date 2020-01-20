import os.path
import single,files

def frame_feats(frame_path,model_path,out_path=None):
    if(not out_path):
        dst_dir=os.path.split(model_path)[0]
        out_path=dst_dir+'/seqs'	
    files.make_dir(out_path)
    for i,in_i in enumerate(files.top_files(model_path)):
        out_i="%s/nn%d"%(out_path,i)
        single.extract_frame_feats(frame_path,in_i,out_i)

def ts_feats(seqs_path,out_path=None):
    if(not out_path):
        dst_dir=os.path.split(seqs_path)[0]
        out_path=dst_dir+'/feats'	
    files.make_dir(out_path)
    for i,in_i in enumerate(files.top_files(seqs_path)):
        out_i="%s/nn%d"%(out_path,i)
        single.compute_ts_feats(in_i,out_i)

frame_path='../../sim3/time'
model_path='../../sim3/ens3/models'
#frame_feats(frame_path,model_path)
ts_feats('../../sim3/ens3/seqs')