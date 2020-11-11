import numpy as np
from keras.models import load_model
import files,spline,stats,data
import basic.ts,single

def reg_pretrain(in_path,reg_epochs=3000,clf_epochs=1000):
	post=['seqs','spline','stats','feats','reg_nn','clf_nn']
	paths=files.get_paths(in_path,post)
	spline.ens_upsample(paths['seqs'],paths["spline"])
	stats.ens_stats(paths['seqs'],paths['stats'])

	train_reg(paths['spline'],paths['stats'],paths['reg_nn'],3000)
	pretrain_clf(paths['reg_nn'],paths['spline'],paths['clf_nn'],clf_epochs)
	basic.ts.ens_extract(paths["spline"],paths["clf_nn"],paths["feats"])

def train_reg(seq_path,feat_path,nn_path,n_epochs=1500):
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
	model.fit(X,y,epochs=n_epochs)
	model.save(nn_path)

def pretrain_clf(nn_path,seq_path,clf_path,n_epochs=1000):
	reg_model=load_model(nn_path)
	dims=reg_model.get_input_at(0).shape
	params={'ts_len':int(dims[1]), 'n_feats':int(dims[2]),
			'n_cats':20}
	clf_model=basic.ts.clf_model(params)
	
	for i in range(7):
		print(i)
		weigts_i=reg_model.layers[i].get_weights()
		clf_model.layers[i].set_weights(weigts_i)

	seq_dict=single.read_frame_feats(seq_path)
	train,test=data.split_dict(seq_dict)
	X,y=basic.ts.get_data(train)
	clf_model.fit(X,y,epochs=n_epochs,batch_size=64)
	clf_model.save(clf_path)

reg_pretrain("reg",reg_epochs=3000)

#train_reg("reg/spline","reg/stats","reg/nn")
#pretrain_clf("reg/nn","reg/spline","reg/clf")

#basic.ts.ens_extract("reg/spline","reg/clf","reg/feats")