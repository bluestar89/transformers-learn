from transformers.models.gpt2 import GPT2Model, GPT2Tokenizer
import torch

repo_name = "openai-community/gpt2"
model_name_or_path = r"G:\hugging-face-models\openai-community\gpt2\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
model = GPT2Model.from_pretrained(model_name_or_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

model_inputs = {
    "input_ids": torch.tensor(
        data=[[109432, 104130,   9370,  99584,  99931,  45995]], dtype=torch.long
    ).to(device=model.device),
    "attention_mask": torch.tensor(data=[[1, 1, 1, 1, 1, 1]], dtype=torch.long).to(
        device=model.device
    ),
}
model_outputs = model.fortrward(**model_inputs, use_cache=True)

concat_next_model_outputs = model.forward(
    **{
        "input_ids": torch.tensor(data=[[3837]], dtype=torch.long).to(device=model.device),
        "attention_mask": torch.tensor(data=[[1]], dtype=torch.long).to(device=model.device),
    },
    past_key_values=model_outputs.past_key_values,
)



normal_next_model_inputs = {
    "input_ids": torch.tensor(
        data=[[109432, 104130,   9370,  99584,  99931,  45995, 3837]], dtype=torch.long
    ).to(device=model.device),
    "attention_mask": torch.tensor(data=[[1, 1, 1, 1, 1, 1, 1]], dtype=torch.long).to(
        device=model.device
    ),
}
normal_next_model_outputs = model.forward(**normal_next_model_inputs, use_cache=True)
