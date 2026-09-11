from llama_cpp import Llama

# Load a locally downloaded model
llm = Llama(
    model_path= "/home/yun/dev/uni/220/wk4/llama-3.2-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,  # Context size
    n_gpu_layers=-1  # Use all available GPU layers if supported
)

# Basic text generation
output = llm("The quick brown fox jumps", stop=["."])
print(output["choices"][0]["text"])   

## summary ##
# 1. download model and put in a path that is memorable // useable for your OS
# 2. follow the prompt in "Load a locally downloaded model"
# 3. point the model path to the exact file