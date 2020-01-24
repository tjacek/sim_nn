import numpy as np
import imgs

def outliner_transform(in_path,out_path):
    def helper(frames):
        pclouds=[ nonzero_points(frame_i) for frame_i in frames]
        center=center_of_mass(pclouds)
        pclouds=[ pcloud_i.T-center for pcloud_i in pclouds ]
        pclouds=[ pcloud_i *pcloud_i for pcloud_i in pclouds ]
        raise Exception(center)
    imgs.transform(in_path,out_path,helper,single_frame=False)

def nonzero_points(frame_i):
    xy_nonzero=np.nonzero(frame_i)
    z_nozero=frame_i[xy_nonzero]
    xy_nonzero,z_nozero=np.array(xy_nonzero),np.expand_dims(z_nozero,axis=0)
    return np.concatenate([xy_nonzero,z_nozero],axis=0)

def center_of_mass(pclouds):
    all_points=[]
    for pcloud_i in pclouds:
        for point_j in pcloud_i.T:
            all_points.append(point_j)
    return np.mean(all_points,axis=0)

outliner_transform("../MSR_out/box","../MSR_out/imgset")