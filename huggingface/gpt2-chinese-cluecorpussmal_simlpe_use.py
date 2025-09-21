from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
# 使用绝对路径，否则会先去huggingface下载
# 目录是包含config.json的目录

model_path = r"G:\hugging-face-models\uer\gpt2-chinese-cluecorpussmall\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3"
# 加载模型和分词器

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 使用本地模型和分词器创建文本生成器

generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device="cpu")
output = generator(
    "今天天气真好啊",  # 生成文本的输入种子文本（prompt）。模型会根据这个初始文本，生成后续的文本
    max_length=50,  # 指定生成文本的最大长度。这里的50表示生成的文本最多包含50个标记（tokens）
    num_return_sequences=1,  # 参数指定返回多少个独立生成的文本序列。值为1表示只生成并返回一段文本。
    truncation=True,  # 该参数决定是否重新输入文本以适应模型的最大输入长度。如果True，超出模型最大输入长度的部分将被截断；如果False，模型可能无法处理过长的输入
    temperature=0.7,  # 该参数控制生成文本的随机性。值越低，生成的文本越保守（倾向于选择概率较高的词）；值越高，生成的文本越多样（倾向于选择更多不同的词）。0.7是一个常用值
    top_k=50,  # 该参数控制模型在每一步生成时仅从概率最高的k个词中选择下一个词。这里top_k=50表示模型在生成每个词时只使用概率最高的前50个候选词，从而减少生成不合理内容
    top_p=0.9,  # 该参数（又称为核采样）进一步限制模型生成时的词汇选择范围。它会选择一组累积概率达到p的词汇，模型只会从这个概率集合中采样。top_p=0.9意味着模型会在概率最高的词汇中采样
    clean_up_tokenization_spaces=True  # 该参数控制生成的文本中是否清理分词时引入的空格。如果为True，生成的文本会清除多余的空格；如果为False，则保留原样
)

print(output)