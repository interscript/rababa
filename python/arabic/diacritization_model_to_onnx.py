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
max_len = 200 # 600 for the original length
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
                  input_names=['src', 'lengths'],
                  output_names=['output'],
                  dynamic_axes = {'src': [1], #[0,1,2], #[0,1,2],
                  #'input_2':{0:'batch'},
                  'output': [1]
                  })

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

print('outs:: ', ort_outs)
print('src:: ', src.detach().numpy().astype(np.int64))
print('lengths: ',lengths.detach().numpy().astype(np.int64))


for i in range(batch_size):
    np.testing.assert_allclose(torch_out['diacritics'][i].detach().numpy(),
                               ort_outs[0][i], rtol=1e-03, atol=1e-03)

print("\n!!!Exported model has been tested with ONNXRuntime, result looks good within given tolerance!!!")



vec = [[41, 12, 40] for i in range(batch_size)]
src = torch.Tensor(vec).long()

lengths = torch.Tensor([3 for i in range(batch_size)]).long()

ort_inputs = {ort_session.get_inputs()[0].name: src.detach().numpy().astype(np.int64),
              ort_session.get_inputs()[1].name: lengths.detach().numpy().astype(np.int64)}


ort_outs = ort_session.run(None, ort_inputs)

torch_out = dia.model(src, lengths)

for i in range(batch_size):
    np.testing.assert_allclose(torch_out['diacritics'][i].detach().numpy(), \
                               ort_outs[0][i], rtol=1e-03, atol=1e-03)


"""
    Test ONNX model on randomized data
"""

import random
test_id = 0

print('***** Test MAX size :: Random Boolean vectors: *****')

for test_run in range(3):

    vec = [[random.randint(0,1) for i in range(max_len)]
            for i in range(batch_size)]
    src = torch.Tensor(vec).long()
    lengths = torch.Tensor([max_len for i in range(batch_size)]).long()

    torch_out = dia.model(src, lengths)
    
    # prepare onnx input
    ort_inputs = {ort_session.get_inputs()[0].name: src.detach().numpy().astype(np.int64),
                  ort_session.get_inputs()[1].name: lengths.detach().numpy().astype(np.int64)}

    # run onnx model
    ort_outs = ort_session.run(None, ort_inputs)

    for i in range(batch_size):
        np.testing.assert_allclose(torch_out['diacritics'][i].detach().numpy(), \
                                   ort_outs[0][i], rtol=1e-03, atol=1e-03)

    print('test :: ', test_run)
    print("Result looks good within given tolerance!!!")


print('***** Test MAX size :: Random float, vectors within 0:16 *****')

for test_run in range(3):

    vec = [[random.randint(0, 17) for i in range(max_len)]
            for i in range(batch_size)]
    src = torch.Tensor(vec).long()
    torch_out = dia.model(src, lengths)

    #my_list = torch_out['diacritics'].detach().numpy().tolist()
    # prepare onnx input
    ort_inputs = {ort_session.get_inputs()[0].name: src.detach().numpy().astype(np.int64),
                  ort_session.get_inputs()[1].name: lengths.detach().numpy().astype(np.int64)}

    # run onnx model
    ort_outs = ort_session.run(None, ort_inputs)

    for i in range(batch_size):
        np.testing.assert_allclose(torch_out['diacritics'][i].detach().numpy(), \
                                   ort_outs[0][i], rtol=1, atol=1)

    print('test :: ', test_run)
    print("Result looks good within given tolerance!!!")


print('***** Test Dynamical sizes :: Random Boolean vectors: *****')

for l in [2, 10, 40, 100, 150]:

    print('length:: ', l)

    vec = [[1 for i in range(l)] # random.randint(0,1)
            for i in range(batch_size)]
    src = torch.Tensor(vec).long()
    lengths = torch.Tensor([l for i in range(batch_size)]).long()

    torch_out = dia.model(src, lengths)

    # prepare onnx input
    ort_inputs = {ort_session.get_inputs()[0].name: src.detach().numpy().astype(np.int64),
                  ort_session.get_inputs()[1].name: lengths.detach().numpy().astype(np.int64)}

    # run onnx model
    ort_outs = ort_session.run(None, ort_inputs)

    for i in range(batch_size):
        np.testing.assert_allclose(torch_out['diacritics'][i].detach().numpy(), \
                                   ort_outs[0][i], rtol=1e-03, atol=1e-03)

    print('test :: ', l)
    print("Result looks good within given tolerance!!!")


print('***** Test Dynamical sizes :: Random float, vectors within 0:16 *****')

for l in [2, 10, 40, 100, 150]:

    vec = [[random.randint(0, 17) for i in range(l)]
            for i in range(batch_size)]
    src = torch.Tensor(vec).long()
    lengths = torch.Tensor([l for i in range(batch_size)]).long()

    torch_out = dia.model(src, lengths)

    # prepare onnx input
    ort_inputs = {ort_session.get_inputs()[0].name: src.detach().numpy().astype(np.int64),
                  ort_session.get_inputs()[1].name: lengths.detach().numpy().astype(np.int64)}

    # run onnx model
    ort_outs = ort_session.run(None, ort_inputs)

    for i in range(batch_size):
        np.testing.assert_allclose(torch_out['diacritics'][i].detach().numpy(), \
                                   ort_outs[0][i], rtol=1, atol=1)

    print('test :: ', l)
    print("Result looks good within given tolerance!!!")
