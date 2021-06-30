
import torch
import torch.nn as nn

from embeddings import (letterToIndex, letterToTensor, lineToTensor)
from model import RNN

# from __future__ import unicode_literals, print_function, division
from io import open
import numpy as np
import glob
import os
import pickle


######################################################################
#        Load Embedding Data & Model                                 #
######################################################################

# load model
model_name_path = '../models-data/name-classification.pt'
n_letters, n_hidden, n_categories = 57, 128, 18 # need to be loaded...
rnn = torch.load(model_name_path)

# load embedding data
with open('../data/dictionary_data.pkl', 'rb') as handle:
    dic_data = pickle.load(handle)
all_categories = dic_data['all_categories']
category_lines = dic_data['category_lines']
all_letters = dic_data['all_letters']
n_letters = len(all_letters)

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line, all_letters, n_letters))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

            
#predict('Dovesky')
#predict('Jackson')
predict('Satoshi')


######################################################################
#        Model to ONNX                                               #
######################################################################

import onnx
import onnxruntime


onnx_model_filename = '../models-data/name_classif_model.onnx'

# random data
hidden = torch.zeros(1, n_hidden)
input = lineToTensor('Albert', all_letters, n_letters)
output, next_hidden = rnn(input[0], hidden)
input_data = input[0], hidden

torch.onnx.export(rnn, input_data, onnx_model_filename, verbose=True)
print('Model printed in rel. path:', onnx_model_filename)


######################################################################
#        Load ONNX Model, predict and test                           #
######################################################################

onnx_model = onnx.load(onnx_model_filename)
onnx.checker.check_model(onnx_model)
print('print graph: ', onnx.helper.printable_graph(onnx_model.graph))

### Load onnx file
ort_session = onnxruntime.InferenceSession(onnx_model_filename)

ort_inputs = {ort_session.get_inputs()[0].name: input_data[0].detach().numpy(),
              ort_session.get_inputs()[1].name: input_data[1].detach().numpy()}

ort_outs = ort_session.run(None, ort_inputs)


### Test onnx computations
torch_out = rnn(input[0], hidden)#input_data[0].detach().numpy(), input_data[1].detach().numpy())
np.testing.assert_allclose(torch_out[0][0].detach().numpy(), ort_outs[0][0], rtol=1e-02, atol=1e-02)
np.testing.assert_allclose(torch_out[1][0].detach().numpy(), ort_outs[1][0], rtol=1e-02, atol=1e-02)
print("Exported model has been tested with ONNXRuntime, and the result looks good!!!")

