import numpy as np
import scipy.signal
from scipy.interpolate import CubicSpline
import files,single,ens

class SplineUpsampling(object):
    def __init__(self,new_size=128):
        self.name="spline"
        self.new_size=new_size

    def __call__(self,feat_i):
        old_size=feat_i.shape[0]
        old_x=np.arange(old_size)
        old_x=old_x.astype(float)  
        if(self.new_size):
            step=float(self.new_size)/float(old_size)
            old_x*=step     
            cs=CubicSpline(old_x,feat_i)
            new_size=np.arange(self.new_size)  
            return cs(new_size)
        else:
            cs=CubicSpline(old_x,feat_i)
            return cs(old_x)

def ens_upsample(in_path,out_path,size=64):
    ens.transform_template(upsample,in_path,out_path,size)

def upsample(in_path,out_path,size=64):
    seq_dict=single.read_frame_feats(in_path)   
    spline=SplineUpsampling(size)
    seq_dict={ name_i:spline(seq_i) for name_i,seq_i in seq_dict.items()}
    single.save_frame_feats(seq_dict,out_path)

if __name__ == "__main__":
    ens_upsample("../ens/seqs",'../ens/spline')