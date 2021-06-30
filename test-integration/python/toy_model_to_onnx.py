# https://github.com/onnx/tutorials/blob/master/tutorials/PytorchOnnxExport.ipynb

import numpy as np
import torch
import torch.nn as nn
from toy_model import toyNNets


######################################################################
#        Build Models                                                #
######################################################################

n_letters, n_hidden, n_categories = 57, 128, 18


rnn = toyNNets(n_letters, n_hidden, n_categories)


input_data = torch.from_numpy(np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0.]], dtype=np.float64)).float()

a = rnn(input_data)


######################################################################
#        Export Models                                               #
######################################################################

onnx_model_filename = '../models-data/toy_model.onnx'

# torch.onnx.export(rnn, input, onnx_model_filename, verbose=True)
torch.onnx.export(rnn,               # model being run
                  input_data,                         # model input (or a tuple for multiple inputs)
                  onnx_model_filename,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'output' : {0 : 'batch_size'},    # variable length axes
                                'hidden' : {0 : 'batch_size'}})
print('Model printed in rel. path:', onnx_model_filename)


######################################################################
#        Load Model, predict and test                                #
######################################################################

import onnx
import onnxruntime


onnx_model = onnx.load(onnx_model_filename)
onnx.checker.check_model(onnx_model)
print('print graph: ', onnx.helper.printable_graph(onnx_model.graph))


### Load onnx file
ort_session = onnxruntime.InferenceSession(onnx_model_filename)

ort_inputs = {ort_session.get_inputs()[0].name: input_data.detach().numpy()} 
ort_outs = ort_session.run(None, ort_inputs)


### Test onnx computations
torch_out = rnn(input_data)
np.testing.assert_allclose(torch_out[0].detach().numpy(), ort_outs[0][0], rtol=1e-03, atol=1e-03)
print("Exported model has been tested with ONNXRuntime, and the result looks good!!!")

