const home = process.env.USERPROFILE || 'C:/Users/sylva';
const defaultLlamaDir = `${home}/apps/llama-b70-sycl`;

export const config = {
  port: Number(process.env.LLAMA_MANAGER_PORT || 31337),
  llamaDir: process.env.LLAMA_CPP_DIR || defaultLlamaDir,
  llamaServer: process.env.LLAMA_SERVER_EXE || `${defaultLlamaDir}/llama-server.exe`,
  oneApiSetvars: process.env.ONEAPI_SETVARS || 'C:/Program Files (x86)/Intel/oneAPI/setvars.bat',
  huggingFaceCli: process.env.HUGGINGFACE_CLI || 'C:/Program Files/Python310/Scripts/huggingface-cli.exe',
  hfCache: process.env.HF_HUB_CACHE || `${home}/.cache/huggingface/hub`,
  defaultModelHint: process.env.LLAMA_DEFAULT_MODEL_HINT || 'Qwen_Qwen3.6-27B-Q6_K.gguf',
};
