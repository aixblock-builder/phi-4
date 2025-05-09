# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams

# max_model_len, tp_size = 8192, 2
# model_name = "tonyshark/deepseek-v3-1b"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
# sampling_params = SamplingParams(temperature=0.3, max_tokens=256, stop_token_ids=[tokenizer.eos_token_id])

# messages_list = [
#     [{"role": "user", "content": "Who are you?"}],
#     [{"role": "user", "content": "Translate the following content into Chinese directly: DeepSeek-V2 adopts innovative architectures to guarantee economical training and efficient inference."}],
#     [{"role": "user", "content": "Write a piece of quicksort code in C++."}],
# ]

# prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]

# outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

import torch

# generated_text = [output.outputs[0].text for output in outputs]
# print(generated_text)
from auto_round import AutoRoundConfig  # #must import for autoround format
from transformers import AutoModelForCausalLM, AutoTokenizer

quantized_model_dir = "OPEA/DeepSeek-V3-int4-sym-inc-cpu"
quantization_config = AutoRoundConfig(backend="cpu")
model = AutoModelForCausalLM.from_pretrained(
    quantized_model_dir,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cpu",
    revision="8fe0735",  ##use autoround format, the only difference is config.json
    quantization_config=quantization_config,  ##cpu only machine don't need to set this value
)

tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, trust_remote_code=True)
prompt = "There is a girl who likes adventure,"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=200,  ##change this to align with the official usage
    do_sample=False,  ##change this to align with the official usage
)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
