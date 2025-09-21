from transformers import AutoModel, AutoTokenizer

model_name = "uer/gpt2-chinese-cluecorpussmall"
model_path = r"G:\hugging-face-models\uer\gpt2-chinese-cluecorpussmall"
# 下载模型到本地
model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
# 下载tokenizer分词器到本地
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
print("模型下载完成")
