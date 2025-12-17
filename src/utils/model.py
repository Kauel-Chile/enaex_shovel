import onnxruntime as ort

def load_model(path_model: str):    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"] 
    return ort.InferenceSession(path_model, sess_options, providers=providers)