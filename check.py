from keras.models import load_model
from keras.models import load_model
import files

def check_model(in_path):
    model_path= in_path+"/frame_models"
    paths=files.top_files(model_path)
    model=load_model(paths[0])
    model.summary()

check_model("../ens6")
