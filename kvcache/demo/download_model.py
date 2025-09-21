from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer
from transformers import AutoModel, AutoTokenizer

# model_name = "Qwen/Qwen2.5-0.5B"
# model_path = r"G:\hugging-face-models\Qwen\Qwen2.5-0.5B"
#
# # 下载模型到本地
# model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
# # 下载tokenizer分词器到本地
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
#


model_name = "openai-community/gpt2"
model_path = r"G:\hugging-face-models\openai-community\gpt2"

# 下载模型到本地
model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
# 下载tokenizer分词器到本地
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)

