import os

import torch
import wandb
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,  # BitsAndBytesConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from huggingface_hub.hf_api import HfFolder

HfFolder.save_token("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")
wandb.login("allow", "69b9681e7dc41d211e8c93a3ba9a6fb8d781404a")
model_id = "tonyshark/llama-3.2-nemotron-3b-instruct"  # mistralai/Mistral-Nemo-Instruct-2407 nvidia/Mistral-NeMo-Minitron-8B-Instruct https://huggingface.co/tonyshark/llama-3.2-nemotron-3b-instruct-ONNX-INT4 tonyshark/llama-3.2-nemotron-3b-instruct itsnebulalol/Llama-3.2-Nemotron-3B-Instruct tiiuae/Falcon3-1B-Instruct google/flan-t5-small meta-llama/Llama-3.2-1B-Instruct smallstepai/Misal-1B-base-v0.1 NickyNicky/experimental-Mistral-1b-V00 hanane22/falcon-1b-instruct-ft Qwen/Qwen2.5-1.5B
output_dir = "./training_output"
repo_id = "llama-3.2-nemotron-3b-instruct"

from_pretrained_kwargs = {
    # "torch_dtype": getattr(torch, "bfloat16"),
    # "trust_remote_code": True,
    # "cache_dir": '',
    # "attn_implementation": "flash_attention_2",
    # "device_map": "auto",
}
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=False,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=False,
# )
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)

# chat_template = {"role": "system", "content": "You are a helpful assistant."}
# chat_template = [
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": prompt}
#                 ]
# tokenizer.apply_chat_template = chat_template
# if '<unk>' in tokenizer.get_vocab():
#     tokenizer.pad_token = '<unk>'
# else:
tokenizer.pad_token = tokenizer.eos_token

# # Update pad token id in model and its config
model.pad_token_id = tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA configuration and application
# use_lora = False
# if use_lora:
#     lora_config = LoraConfig(
#         r=32,
#         lora_alpha=8,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
#         lora_dropout=0.1,
#         bias="none",
#         task_type="CAUSAL_LM",
#         modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"]
#     )

#     model = get_peft_model(model, lora_config)

# Dataset loading
train_dataset = load_dataset("goutampunasiya/pretraining-data-stories-input-ids")
# train_dataset = train_dataset.rename_column('chosen', 'messages')

training_args = TrainingArguments(
    # max_steps=1,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    bf16=True,
    learning_rate=1e-4,
    lr_scheduler_type="constant",
    save_steps=0,
    optim="adamw_torch",
    save_strategy="steps",
    logging_dir=output_dir + "/logs",
    output_dir=output_dir,
    warmup_ratio=0.03,
    logging_steps=1,
    # hub_private_repo=True,
    remove_unused_columns=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    save_total_limit=1,
    report_to="wandb",
    push_to_hub=True,
    hub_model_id=repo_id,
    hub_token="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",
)
from transformers import TrainerCallback


class TrainOnStartCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, logs=None, **kwargs):
        # Log training loss at step 0
        logs = logs or {}
        self.log(logs)

    def log(self, logs):
        print(f"Logging at start: {logs}")


model.config.use_cache = True
trainer = SFTTrainer(
    max_seq_length=2048,
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    # dataset_text_field="text",
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False,  # No need to add additional separator token
        "skip_prepare_dataset": True,  # skip the dataset preparation
    },
    callbacks=[TrainOnStartCallback()],
)

# Training

trainer.train()
trainer.push_to_hub(
    # repo_id=repo_id,
    # commit_message=repo_id,
    # private=False,
    # branch="main",
    # create_pr=True,
    # token="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"
)
# Define the save and push paths
# new_model_repo = repo_id
# local_save_path_model = f"{new_model_repo}-local"

# trainer.save_model(local_save_path_model)
# tokenizer.save_pretrained(local_save_path_model)
