import os,re,os.path

def top_files(path):
    paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    paths=sorted(paths,key=natural_keys)
    return paths

def bottom_files(path,full_paths=True):
    all_paths=[]
    for root, directories, filenames in os.walk(path):
        if(not directories):
            for filename_i in filenames:
                path_i= root+'/'+filename_i if(full_paths) else filename_i
                all_paths.append(path_i)
    all_paths.sort(key=natural_keys)        
    return all_paths

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def clean_str(name_i):
    name_i=name_i.split("/")[-1]
    digits=[ str(int(digit_i)) for digit_i in re.findall(r'\d+',name_i)]
    return "_".join(digits)

def clean_dict(dict_i):
    return {clean_str(name_j):seq_j  
                for name_j,seq_j in dict_i.items()}

def prepare_dirs(in_path,name,sufixes):
    dir_path=os.path.dirname(in_path)
    if(name):
        dir_path="%s/%s" %(dir_path,name)
        make_dir(dir_path)
    return get_paths(dir_path,sufixes)  

def get_paths(dir_path,sufixes):
    return {sufix_i:"%s/%s"%(dir_path,sufix_i) for sufix_i in sufixes }