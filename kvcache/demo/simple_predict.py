from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer

repo_name = "Qwen/Qwen2.5-0.5B"
model_name_or_path = r"G:\hugging-face-models\Qwen\Qwen2.5-0.5B\models--Qwen--Qwen2.5-0.5B\snapshots\060db6499f32faf8b98477b0a26969ef7d8b9987"
model = Qwen2ForCausalLM.from_pretrained(model_name_or_path)
tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path)

text = "介绍一下杭州的良脑路"

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
for k, v in model_inputs.items():
    print(k, v)

generated_ids = model.generate(**model_inputs, max_new_tokens=10)

# for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
#     print("input_ids = ", input_ids)
#     print("output_ids = ", output_ids)

generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)