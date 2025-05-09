import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from huggingface_hub.hf_api import HfFolder

HfFolder.save_token("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")
output_dir = "./data/checkpoint"

from_pretrained_kwargs = {
    # "torch_dtype": getattr(torch, "bfloat16"),
    "trust_remote_code": True,
    "cache_dir": "",
    "trust_remote_code": True,
    # "attn_implementation": "flash_attention_2",
    # "device_map": "auto",
}
from inference.configuration_deepseek import DeepseekV3Config
from inference.modeling_deepseek import DeepseekV3ForCausalLM

# Initializing a Deepseek-V3 style configuration
configuration = DeepseekV3Config()

# from transformers import AutoTokenizer
model = DeepseekV3ForCausalLM.from_pretrained("tonyshark/deepseek-v3-1b")
tokenizer = AutoTokenizer.from_pretrained("tonyshark/deepseek-v3-1b")
# Accessing the model configuration
configuration = model.config
print(configuration)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)
result = tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(result)

# model = AutoModelForCausalLM.from_pretrained(
#     "tonyshark/deepseek-v3-1b",
#     **from_pretrained_kwargs
# )

# tokenizer = AutoTokenizer.from_pretrained("tonyshark/deepseek-v3-1b", trust_remote_code=True)

if "<unk>" in tokenizer.get_vocab():
    tokenizer.pad_token = "<unk>"
else:
    tokenizer.pad_token = tokenizer.eos_token

# Update pad token id in model and its config
model.pad_token_id = tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA configuration and application
use_lora = True
if use_lora:
    lora_config = LoraConfig(
        r=32,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=[
            "embed_tokens",
            "input_layernorm",
            "post_attention_layernorm",
            "norm",
        ],
    )

    model = get_peft_model(model, lora_config)

# Dataset loading
train_dataset = load_dataset("timdettmers/openassistant-guanaco", split="train[:100]")
train_dataset = train_dataset.rename_column("chosen", "messages")

training_args = TrainingArguments(
    max_steps=1,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    bf16=True,
    learning_rate=1e-4,
    lr_scheduler_type="constant",
    save_steps=0,
    # optim="adamw_torch",
    save_strategy="steps",
    logging_dir=output_dir + "/logs",
    output_dir=output_dir,
    warmup_ratio=0.03,
    logging_steps=1,
    hub_private_repo=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    save_total_limit=1,
)

trainer = SFTTrainer(
    max_seq_length=2048,
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
)

# Training
model.config.use_cache = False
trainer.train()

# Define the save and push paths
new_model_repo = f"tonyshark/fine-tuned-deepseek-v3"
local_save_path_model = f"{new_model_repo}-local"

trainer.save_model(local_save_path_model)
tokenizer.save_pretrained(local_save_path_model)
