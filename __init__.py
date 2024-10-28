from inference import model2annotations,traverse_by_dict, init_model
import os

def init():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = f'{current_directory}/data/comictextdetector.pt.onnx'
    return init_model(model_path, device = 'cpu')
def inference(model):
    #model_path = f'{current_directory}/data/comictextdetector.pt'
    
    # todo 路径
    img_dir = '/home/ubuntu/translation/input'
    save_dir = '/home/ubuntu/translation/output'
    model2annotations(img_dir, save_dir, save_json=True, model=model)
    return traverse_by_dict(img_dir, save_dir)