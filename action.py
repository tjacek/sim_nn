import frames.action,exp, frames.new

def exp1(in_path,n_epochs=350):
	paths=exp.get_paths(in_path,["frames","nn","feat"])
	print(paths)
	frames.action.make_model(paths["frames"],paths["nn"],n_epochs)
	frames.action.extract(paths["nn"],paths["feat"],paths["frames"])

def ens_exp(in_path):
    paths=exp.get_paths(in_path,["frames","models","feats"])
#    frames.action.ens_train(paths["frames"],paths["models"],n_epochs=350)
    frames.action.ens_extract(paths["frames"],paths["models"], paths["feats"])

def ens_test(in_path):
    paths=exp.get_paths(in_path,["frames","models","feats"])
#    sim_frames=frames.new.get_sim_frames()
#    sim_frames.ens_train(paths["frames"],paths["models"],n_epochs=350)
    frames.ens_extract(paths["frames"],paths["models"], paths["seqs"])

exp1("../simple/fore/exp2")
#ens_test("../simple/bound/exp3")