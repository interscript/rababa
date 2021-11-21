import torch
import pickle
import random

import torch
import onnx
import onnxruntime

import numpy as np

from diacritizer import Diacritizer


"""
    Key Params:
        max_len:
            is the max length for the arabic strings to be diacritized
        batch size:
            has to do with the model training and usage
"""

d_params = yaml.load(open("config/convert_torch_onnx.yml"))
max_len = d_params["max_len"]  # 600 for the original length
batch_size = d_params["batch_size"]
config_str = d_params["config_str"]
model_kind_str = d_params["model_kind_str"]
onnx_model_filename = d_params["onnx_model_filename"]
device = d_params["device"]


"""
    example and mock data:
    we found that populating all the data, removing the zeros gives better results.
"""

normalized = torch.Tensor(
    [[1 for i in range(max_len)] for i in range(batch_size)]
).long()


"""
    Instantiate Diacritization model
"""

dia = Diacritizer(config_str, model_kind_str, True)

# set model to inference mode
dia.model.to(device)
dia.model.eval()
normalized = normalized.to(device)

# run model
niqqud, dagesh, sin = dia.model(normalized)
torch_outs = dia.model(normalized)  # niqqud, dagesh, sin


"""
    Load ONNX libs and export models into onnx
"""

onnx_model_filename = "../models-data/diacritization_model.onnx"

# export model
torch.onnx.export(
    dia.model,
    normalized,
    onnx_model_filename,
    verbose=False,
    opset_version=11,
    input_names=["normalized"],
    output_names=["niqqud", "dagesh", "sin"],
    dynamic_axes={"normalized": [1], "output": [1], "dagesh": [1], "sin": [1]},
)

print("Model printed in rel. path:", onnx_model_filename)


"""
    Load ONNX versions of model
"""

# load model
onnx_model = onnx.load(onnx_model_filename)
# check model
onnx.checker.check_model(onnx_model)
# inference session
ort_session = onnxruntime.InferenceSession(onnx_model_filename)

# get onnx inputs and outputs names
# ort_session.get_inputs(), ort_session.get_outputs()


"""
    Run ONNX model on sample data
"""

# prepare onnx input
ort_inputs = {
    ort_session.get_inputs()[0].name: normalized.detach().numpy().astype(np.int64)
}

# run onnx model
ort_outs = ort_session.run(None, ort_inputs)


for i in range(batch_size):
    for dim in range(3):  # niqqud, dagesh, sin
        np.testing.assert_allclose(
            torch_outs[dim][i].detach().numpy(),
            ort_outs[dim][i],
            rtol=1e-02,
            atol=1e-02,
        )

print(
    "\n!!!Exported model has been tested with ONNXRuntime, \
            result looks good within given tolerance!!!"
)


vec = [[41, 12, 40] for i in range(batch_size)]
normalized = torch.Tensor(vec).long()

ort_inputs = {
    ort_session.get_inputs()[0].name: normalized.detach().numpy().astype(np.int64)
}


"""
    Test ONNX model on randomized data
"""

test_id = 0

print("***** Test MAX size :: Random Boolean vectors: *****")
print(max_len)

for test_run in range(3):

    vec = [[random.randint(0, 1) for i in range(max_len)] for i in range(batch_size)]
    normalized = torch.Tensor(vec).long()

    torch_outs = dia.model(normalized)
    # prepare onnx input
    ort_inputs = {
        ort_session.get_inputs()[0].name: normalized.detach().numpy().astype(np.int64)
    }

    # run onnx model
    ort_outs = ort_session.run(None, ort_inputs)

    for i in range(batch_size):
        for dim in range(3):
            np.testing.assert_allclose(
                torch_outs[dim][i].detach().numpy(),
                ort_outs[dim][i],
                rtol=1e-01,
                atol=1e-01,
            )

    print("test :: ", test_run)
    print("Result looks good within given tolerance!!!")


print("***** Test MAX size :: Random float, vectors within 0:16 *****")
print(max_len)

for test_run in range(3):

    vec = [[random.randint(0, 17) for i in range(max_len)] for i in range(batch_size)]
    normalized = torch.Tensor(vec).long()
    torch_out = dia.model(normalized)

    # prepare onnx input
    ort_inputs = {
        ort_session.get_inputs()[0].name: normalized.detach().numpy().astype(np.int64)
    }

    # run onnx model
    ort_outs = ort_session.run(None, ort_inputs)

    for i in range(batch_size):
        for dim in range(3):
            np.testing.assert_allclose(
                torch_out[dim][i].detach().numpy(), ort_outs[dim][i], rtol=1, atol=1
            )

    print("test :: ", test_run)
    print("Result looks good within given tolerance!!!")


print("***** Test Dynamical sizes :: Random Boolean vectors: *****")

for l in [2, 10, 40, 100, 150]:

    print("length:: ", l)

    vec = [[1 for i in range(l)] for i in range(batch_size)]  # random.randint(0,1)
    normalized = torch.Tensor(vec).long()

    torch_out = dia.model(normalized)

    # prepare onnx input
    ort_inputs = {
        ort_session.get_inputs()[0].name: normalized.detach().numpy().astype(np.int64)
    }

    # run onnx model
    ort_outs = ort_session.run(None, ort_inputs)

    for i in range(batch_size):
        for dim in range(3):
            np.testing.assert_allclose(
                torch_out[dim][i].detach().numpy(),
                ort_outs[dim][i],
                rtol=1e-02,
                atol=1e-02,
            )

    print("test :: ", l)
    print("Result looks good within given tolerance!!!")


print("***** Test Dynamical sizes :: Random float, vectors within 0:16 *****")

for l in [2, 10, 40, 100, 150]:

    vec = [[random.randint(0, 17) for i in range(l)] for i in range(batch_size)]
    normalized = torch.Tensor(vec).long()

    torch_out = dia.model(normalized)

    # prepare onnx input
    ort_inputs = {
        ort_session.get_inputs()[0].name: normalized.detach().numpy().astype(np.int64)
    }

    # run onnx model
    ort_outs = ort_session.run(None, ort_inputs)

    for i in range(batch_size):
        for dim in range(3):
            np.testing.assert_allclose(
                torch_out[dim][i].detach().numpy(), ort_outs[dim][i], rtol=1, atol=1
            )

    print("test :: ", l)
    print("Result looks good within given tolerance!!!")
