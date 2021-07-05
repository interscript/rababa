
import pickle
import numpy as np


from diacritizer import CBHGDiacritizer


"""
    Instantiate Diacritization model
"""
model_kind_str = 'cbhg'
config_str = 'config/cbhg.yml'
load_model = True

dia = CBHGDiacritizer(config_str, model_kind_str, load_model)

# set model to inference mode
dia.model.to(dia.device)
dia.model.eval()

# test
#tmp = dia.model(src=batch_data["src"],
#                target=batch_data["target"],
#                lengths=batch_data["lengths"])


"""
    Load sample data to "teach onnx what the computational flow is"
"""

batch_data = pickle.load( open('../data/batch_data.pkl', 'rb') )


"""
    Load ONNX libs and export models into onnx
"""

import torch
import onnx
import onnxruntime


onnx_model_filename = '../models-data/diacritization_model.onnx'

# input data for "teaching onnx about the nnets graph"
src, lengths, target = (batch_data["src"], batch_data["lengths"], batch_data["target"])

# export model
torch.onnx.export(dia.model, (src, lengths, target), onnx_model_filename, verbose=False, opset_version=11, input_names=['src', 'lengths'])
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

# run torch model
torch_out = dia.model(src, lengths)
torch_out['diacritics'][0] # .shape

# comparisons, we compare first 3 vectors for now
np.testing.assert_allclose(torch_out['diacritics'][0].detach().numpy(), ort_outs[0][0], rtol=1e-05, atol=1e-05)
np.testing.assert_allclose(torch_out['diacritics'][1].detach().numpy(), ort_outs[0][1], rtol=1e-05, atol=1e-05)
np.testing.assert_allclose(torch_out['diacritics'][2].detach().numpy(), ort_outs[0][2], rtol=1e-05, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!!!")

