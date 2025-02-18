import bentoml

model1 = bentoml.onnx.get('synthsegrobust2_trace').load_model(providers=['CUDAExecutionProvider'])