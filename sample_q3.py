"""
Sample from a trained model with Radix KV cache
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import time

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'out'

num_samples = 1
max_new_tokens = 200
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Large shared system prompt (~150 tokens)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an advanced language model trained to write detailed, coherent,
and stylistically consistent fantasy narratives. You are especially good
at continuing stories that share a common opening, tone, and worldbuilding.
You should reuse context efficiently, avoid contradictions, and produce
rich descriptions of characters, settings, and events. Always maintain
a consistent narrative voice and avoid repeating the same phrases too often.
The story should unfold naturally and creatively.
""".strip()

SYSTEM_PROMPT *= 2

PROMPTS = [
    SYSTEM_PROMPT + "\n\nOnce upon a time in a distant kingdom, there lived a wise old king",
    SYSTEM_PROMPT + "\n\nOnce upon a time in a distant kingdom, there lived a brave young knight",
    SYSTEM_PROMPT + "\n\nOnce upon a time in a distant kingdom, there lived a clever merchant",
    SYSTEM_PROMPT + "\n\nOnce upon a time in a distant kingdom, the people believed that magic",
    SYSTEM_PROMPT + "\n\nOnce upon a time in a distant kingdom, the people feared the return of",
]

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval().to(device)

if compile:
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# -----------------------------------------------------------------------------
# Run generation
# -----------------------------------------------------------------------------
start_time = time.time()

with torch.no_grad():
    with ctx:
        xes = []
        for i, prompt in enumerate(PROMPTS):
            start_ids = encode(prompt)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, :]
            xes.append(x)

        print(f"\nStarting batch generation with {len(PROMPTS)} prompts...")
        outputs = model.generate_batch_with_radix(
            batch_prompts=xes,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

end_time = time.time()
print(f"\n=== Total Execution Time ===")
print(f"Total time: {end_time - start_time:.2f}s")

# Display the generated outputs
print(f"\n=== Generated Outputs ===")
for i, output in enumerate(outputs):
    # Get the full sequence (prompt + generated)
    full_sequence = output[0].tolist()
    
    # Get the prompt tokens (original input)
    prompt_tokens = xes[i][0].tolist()
    prompt_length = len(prompt_tokens)
    
    # Separate prompt and generated parts
    generated_tokens = full_sequence[prompt_length:]
    
    # Decode both parts
    prompt_text = decode(prompt_tokens)
    generated_text = decode(generated_tokens)
    
    print(f"\nPrompt {i+1}:")
    print(f"  Input: '{prompt_text}'")
    print(f"  Generated: '{generated_text}'")
    print(f"  Total tokens: {len(full_sequence)} (prompt: {prompt_length}, generated: {len(generated_tokens)})")
    
    # Optionally, show the full output
    full_text = decode(full_sequence)
    print(f"  Full output: '{full_text}'")
    print("-" * 80)