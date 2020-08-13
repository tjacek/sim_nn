import frames.action,exp

def exp1(in_path,n_epochs=350):
	paths=exp.get_paths(in_path,["frames","nn","feat"])
	print(paths)
	frames.action.make_model(paths["frames"],paths["nn"],n_epochs)
	frames.action.extract(paths["frames"],paths["nn"],paths["feat"])

#exp1("../MSR_exp1")
#frames.action.ens_train("../MSR_exp1/exp1/frames","../MSR_exp1/exp1/models",n_epochs=350)
frames.action.ens_extract("../MSR_exp1/exp1/frames","../MSR_exp1/exp1/ens/models","../MSR_exp1/exp1/ens/feats")