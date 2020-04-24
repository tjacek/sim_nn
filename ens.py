import os.path
import single,files,basic

def train_template(fun,in_path,out_path,n_epochs=5):
    if(not is_ens( in_path)):
        return fun(in_path,out_path,n_epochs)
    files.make_dir(out_path)
    for i in range(20):
        out_i='%s/nn%d'%(out_path,i)
        fun(in_path,out_i,n_epochs,i)

def extract_template(fun,frame_path,model_path,out_path):
    if(not is_ens( model_path)):
        return fun(frame_path,model_path,out_path)
    files.make_dir(out_path)
    for in_i in files.top_files(frame_path):
        model_i='%s/%s'%(model_path,in_i.split("/")[-1])
        out_i='%s/%s'%(out_path,in_i.split("/")[-1])
        print(in_i)
        fun(in_i,model_i,out_i)

def transform_template(fun,in_path,out_path,info=None):
    if(not is_ens( in_path)):
        return fun(in_path,out_path,info)
    files.make_dir(out_path)
    for in_i in files.top_files(in_path):
        out_i='%s/%s'%(out_path,in_i.split("/")[-1])
        if(info):
            fun(in_i,out_i,info)
        else:
            fun(in_i,out_i)

def is_ens(in_path):
    if(not os.path.isdir(in_path)):
        return False
    return any([  os.path.isdir(file_i)  
                    for file_i in  files.top_files(in_path)] )