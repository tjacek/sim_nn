import frames.action,exp

def exp1(in_path,n_epochs=350):
	paths=exp.get_paths(in_path,["frames","nn","feat"])
	print(paths)
	frames.action.make_model(paths["frames"],paths["nn"],n_epochs)
	frames.action.extract(paths["nn"],paths["feat"],paths["frames"])

def ens_exp(in_path):
    paths=exp.get_paths(in_path,["frames","models","feats"])
    frames.action.ens_train(paths["frames"],paths["models"],n_epochs=350)
    frames.action.ens_extract(paths["frames"],paths["models"], paths["feats"])

exp1("../MSR_exp1/exp5")
#ens_exp("../MSR_exp1/exp2/")