
# Tests for Integration
with ONNX and ruby... mimicking https://github.com/secryst/secryst

Under python there are 2 models:

* toy model:
1. model definition and save /python/toy_model.py
2. model load, translation to onnx and tests /python/toy_model_to_onnx.py
* names classification models
1. model definition /python/model.py
2. data preprocessing and training and save  /python/train.py
3. model load,  to onnx and tests  /python/model_to_onnx.py
