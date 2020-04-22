import os.path 
import files,stats

def stats_feats(in_path):
    dir_path=os.path.dirname(in_path)
    dir_path+="/stats"
    files.make_dir(dir_path)
    stats.ens_stats(in_path,dir_path+"/feats")


stats_feats("../ens2/ts/seqs")