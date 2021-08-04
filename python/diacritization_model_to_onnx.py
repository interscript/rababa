import torch
import pickle

import numpy as np

from diacritizer import Diacritizer


"""
    Key Params:
        max_len: 
            is the max length for the arabic strings to be diacritized
        batch size: 
            has to do with the model training and usage
"""
max_len = 300 # 600 for the original length
batch_size = 32


""" 
    example and mock data:
    we found that populating all the data, removing the zeros gives better results.
"""
src = torch.Tensor([[1 for i in range(max_len)]
                    for i in range(batch_size)]).long()
lengths = torch.Tensor([max_len for i in range(batch_size)]).long()
# example data
batch_data = pickle.load( open('../models-data/batch_example_data.pkl', 'rb') )

#target = batch_data['target']


"""
    Instantiate Diacritization model
"""
model_kind_str = 'cbhg'
config_str = 'config/cbhg.yml'
load_model = True

dia = Diacritizer(config_str, model_kind_str, load_model)

# set model to inference mode
dia.model.to(dia.device)
dia.model.eval();
# run model
torch_out = dia.model(src, lengths)


"""
    Load ONNX libs and export models into onnx
"""

import torch
import onnx
import onnxruntime


onnx_model_filename = '../models-data/diacritization_model.onnx'


# export model
torch.onnx.export(dia.model, 
                  (src, lengths), 
                  onnx_model_filename, 
                  verbose=False, 
                  opset_version=11, 
                  input_names=['src', 'lengths'])
print('Model printed in rel. path:', onnx_model_filename)


"""
    Load ONNX versions of model
"""

# load model
onnx_model = onnx.load(onnx_model_filename)
# check model
onnx.checker.check_model(onnx_model)
# inference session
ort_session = onnxruntime.InferenceSession(onnx_model_filename)

# onnx inputs and outputs names
# ort_session.get_inputs(), ort_session.get_outputs()


"""
    Run ONNX model on sample data
"""

# prepare onnx input
ort_inputs = {ort_session.get_inputs()[0].name: src.detach().numpy().astype(np.int64),
              ort_session.get_inputs()[1].name: lengths.detach().numpy().astype(np.int64)}

# run onnx model
ort_outs = ort_session.run(None, ort_inputs)


for i in range(batch_size):
    np.testing.assert_allclose(torch_out['diacritics'][i].detach().numpy(), ort_outs[0][i], rtol=1e-03, atol=1e-03)


print("\n!!!Exported model has been tested with ONNXRuntime, result looks good within given tolerance!!!")

