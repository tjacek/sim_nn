import os.path
import single,files,basic

def train_template(fun,in_path,out_path,n_epochs=5):
    files.make_dir(out_path)
    for i in range(20):
        out_i='%s/nn%d'%(out_path,i)
        fun(in_path,out_i,n_epochs,cat_i=i)

def extract_template(fun,frame_path,model_path,out_path):
    files.make_dir(out_path)
    for in_i in files.top_files(model_path):
        out_i='%s/nn%s'%(out_path,in_i.split("/")[-1])
        print(in_i)
        fun(frame_path,in_i,out_i)

#def frame_feats(frame_path,model_path,out_path=None,fun=None):
#    if(not out_path):
#        dst_dir=os.path.split(model_path)[0]
#        out_path=dst_dir+'/seqs'	
#    files.make_dir(out_path)
#    if(not fun):
#        fun= single.extract_frame_feats
#    for i,in_i in enumerate(files.top_files(model_path)):
#        out_i="%s/nn%d"%(out_path,i)
#        fun(frame_path,in_i,out_i)

#def ts_feats(seqs_path,out_path=None):
#    if(not out_path):
#        dst_dir=os.path.split(seqs_path)[0]
#        out_path=dst_dir+'/feats'	
#    files.make_dir(out_path)
#    for i,in_i in enumerate(files.top_files(seqs_path)):
#        out_i="%s/nn%d"%(out_path,i)
#        single.compute_ts_feats(in_i,out_i)