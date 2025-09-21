from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer
import torch

repo_name = "Qwen/Qwen2.5-0.5B"
model_name_or_path = r"G:\hugging-face-models\Qwen\Qwen2.5-0.5B\models--Qwen--Qwen2.5-0.5B\snapshots\060db6499f32faf8b98477b0a26969ef7d8b9987"
model = Qwen2ForCausalLM.from_pretrained(model_name_or_path)
tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path)

model_inputs = {
    "input_ids": torch.tensor(
        data=[[109432, 104130,   9370,  99584,  99931,  45995]], dtype=torch.long
    ).to(device=model.device),
    "attention_mask": torch.tensor(data=[[1, 1, 1, 1, 1, 1]], dtype=torch.long).to(
        device=model.device
    ),
}
model_outputs = model.forward(**model_inputs, use_cache=True)
print(model_outputs.keys())
print(model_outputs.logits.shape)
print(model_outputs.logits[:, -1, :].shape)
print(model_outputs.logits[:, -1, :].argmax(dim=-1))
print(tokenizer.decode(model_outputs.logits[:, -1, :].argmax(dim=-1)))


concat_next_model_outputs = model.forward(
    **{
        "input_ids": torch.tensor(data=[[3837]], dtype=torch.long).to(device=model.device),
        "attention_mask": torch.tensor(data=[[1]], dtype=torch.long).to(device=model.device),
    },
    past_key_values=model_outputs.past_key_values,
)
concat_next_model_outputs.keys()
print(concat_next_model_outputs.keys())
print(concat_next_model_outputs.logits.shape)
print(concat_next_model_outputs.logits[:, -1, :].shape)
print(concat_next_model_outputs.logits[:, -1, :].argmax(dim=-1))
print(tokenizer.decode(concat_next_model_outputs.logits[:, -1, :].argmax(dim=-1)))


# normal_next_model_inputs = {
#     "input_ids": torch.tensor(
#         data=[[109432, 104130,   9370,  99584,  99931,  45995, 3837]], dtype=torch.long
#     ).to(device=model.device),
#     "attention_mask": torch.tensor(data=[[1, 1, 1, 1, 1, 1, 1]], dtype=torch.long).to(
#         device=model.device
#     ),
# }
# normal_next_model_outputs = model.forward(**normal_next_model_inputs, use_cache=True)
# normal_next_model_outputs.keys()
# print(normal_next_model_outputs.keys())
# print(normal_next_model_outputs.logits.shape)
# print(normal_next_model_outputs.logits[:, -1, :].shape)
# print(normal_next_model_outputs.logits[:, -1, :].argmax(dim=-1))
# print(tokenizer.decode(normal_next_model_outputs.logits[:, -1, :].argmax(dim=-1)))
