import numpy as np,cv2
import imgs

def outliner_transform(in_path,out_path,dims=0):
    def helper(frames):
        pclouds=[ nonzero_points(frame_i) for frame_i in frames]
        pclouds=normalize(pclouds)
        new_frames=[get_proj(pcloud_i,dims) for pcloud_i in pclouds]        
        action_img=np.mean(new_frames,axis=0)
        action_img[action_img >0]=100
        return action_img
    transform=[helper,smooth_proj,scale]    
    imgs.action_img(in_path,out_path,transform)#,single_frame=False)

def nonzero_points(frame_i):
    xy_nonzero=np.nonzero(frame_i)
    z_nozero=frame_i[xy_nonzero]
    xy_nonzero,z_nozero=np.array(xy_nonzero),np.expand_dims(z_nozero,axis=0)
    return np.concatenate([xy_nonzero,z_nozero],axis=0)

def normalize(pclouds):
    center=center_of_mass(pclouds)
    pclouds=[ pcloud_i.T-center for pcloud_i in pclouds ]
    pclouds=[ pcloud_i *pcloud_i*np.sign(pcloud_i) for pcloud_i in pclouds ]
    pc_min=get_min(pclouds)
    pclouds=[ (pcloud_i-pc_min) for pcloud_i in pclouds]
    pc_max=get_max(pclouds)
    pclouds=[ (pcloud_i/pc_max)*128 for pcloud_i in pclouds]
    return pclouds

def center_of_mass(pclouds):
    all_points=[]
    for pcloud_i in pclouds:
        for point_j in pcloud_i.T:
            all_points.append(point_j)
    return np.mean(all_points,axis=0)

def get_min(pclouds):
    return np.amin([ np.amin(pcloud_i,axis=0) 
                      for pcloud_i in pclouds],axis=0).T

def get_max(pclouds):
    return np.amax([ np.amax(pcloud_i,axis=0) 
                      for pcloud_i in pclouds],axis=0).T

def get_proj(pclouds,dims):
    img_i=np.zeros((128,128),dtype=float)
    x,y=dims,(dims+1)%3
    for point_j in pclouds:
        x_j,y_j=int(point_j[x]),int(point_j[y])
        if( x_j<128  and y_j<128):
            img_i[x_j][y_j]=100
        else:
            print(x_j,y_j)
    return img_i

def smooth_proj(proj_i):
    if(type(proj_i)==list):
        return [smooth_proj(frame_j) for frame_j in proj_i]
    kernel2= np.ones((3,3),np.uint8)
    proj_i = cv2.dilate(proj_i,kernel2,iterations = 1)#
    proj_i[proj_i!=0]=200.0
    return proj_i

def scale(binary_img ,dim_x=64,dim_y=64):
    if(type(binary_img)==list):
        return [  scale(frame_i,dim_x,dim_y) for frame_i in binary_img]
    return cv2.resize(binary_img,(dim_x,dim_y), interpolation = cv2.INTER_CUBIC)

#outliner_transform("../MSR_out/box","../MSR_out/yz",2)
imgs.concat_frames("../MSR_out/xy","../MSR_out/yz","../MSR_out/full")