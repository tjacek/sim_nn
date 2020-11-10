import numpy as np
import files,spline,stats
import basic.ts,single

def reg_pretrain(in_path):
	paths=files.get_paths(in_path,['seqs','spline','stats'])
	spline.ens_upsample(paths['seqs'],paths["spline"])
	stats.ens_stats(paths['seqs'],paths['stats'])
#	print(paths)

def train_reg(seq_path,feat_path):
	lines=open(feat_path,'r').readlines()
	lines=[ line_i.split('#') for line_i in lines]
	feats={ line_i[1]: np.fromstring(line_i[0],sep=",",dtype=float) 
    		for line_i in lines}

	names=list(feats.keys())
	y=np.array([feats[name_i] for name_i in names])

	seq_dict=single.read_frame_feats(seq_path)
	X=np.array([ seq_dict[name_i.strip()]
				for name_i in names])
	params={'ts_len':64, 'n_feats':100}
	model=basic.ts.reg_model(params,n_units=400)
	model.fit(X,y,epochs=300)
#	print(X.shape)		

train_reg("reg/spline","reg/stats")