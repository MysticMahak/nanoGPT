"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import time
from model import GPTConfig, GPT
from typing import Optional, Tuple, List


def find_longest_prefix_match(prompt_tokens: List[int], cached_prompts: List[Tuple[int, ...]]) -> Optional[Tuple[int, Tuple[int, ...]]]:
    """
    Find the longest prefix match between the given prompt and cached prompts.
    
    Args:
        prompt_tokens: List of token IDs for the current prompt
        cached_prompts: List of cached prompt token tuples
    
    Returns:
        Tuple of (prefix_length, cached_prompt_tuple) if a match is found, else None
    """
    if not cached_prompts:
        return None
    
    best_match = None
    best_match_len = 0
    
    for cached_prompt in cached_prompts:
        # Find common prefix length
        prefix_len = 0
        for i in range(min(len(prompt_tokens), len(cached_prompt))):
            if prompt_tokens[i] == cached_prompt[i]:
                prefix_len += 1
            else:
                break
        
        # Update best match if this one is longer
        if prefix_len > best_match_len and prefix_len > 0:
            best_match_len = prefix_len
            best_match = (prefix_len, cached_prompt)
    
    return best_match


def _truncate_past_kv(past_kv, length: int):
    """Return a version of past_kv where each layer's key/value time dimension is truncated to `length`."""
    if past_kv is None:
        return None
    truncated = []
    for layer in past_kv:
        if layer is None:
            truncated.append(None)
            continue
        k, v = layer
        # shapes: (B, nh, T, hs)
        if k is None or v is None:
            truncated.append(None)
        else:
            truncated.append((k[:, :, :length, :].contiguous(), v[:, :, :length, :].contiguous()))
    return truncated


def generate_with_prefix_caching(
    model: GPT,
    prompt_tokens: torch.Tensor,
    prompt_kv_cache: dict,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, dict]:
    """
    Generate tokens with intelligent prefix caching.
    """
    prompt_tuple = tuple(prompt_tokens[0].tolist())
    cached_prompts = list(prompt_kv_cache.keys())
    
    # Try to find a prefix match
    prefix_match = find_longest_prefix_match(list(prompt_tuple), cached_prompts)
    
    past_kv = None
    start_pos = 0  # Renamed from prefix_len to be clearer
    
    if prefix_match is not None:
        prefix_len, matched_cached_prompt = prefix_match
        print(f"Found prefix match: {prefix_len}/{len(prompt_tuple)} tokens")
        
        # Retrieve the cached KV for the matching prefix
        past_kv = prompt_kv_cache[matched_cached_prompt]
        # If the cached prompt is longer than the matching prefix, truncate the cached KV
        if len(matched_cached_prompt) > prefix_len:
            past_kv = _truncate_past_kv(past_kv, prefix_len)
        
        # If the entire prompt is already cached, just use generation with the cache
        if prefix_len == len(prompt_tuple):
            print("Exact prompt match - reusing full KV cache")
            start_pos = prefix_len  # All tokens are already in cache
        else:
            # We need to compute KV for the remaining tokens beyond the prefix
            print(f"Reusing KV cache for first {prefix_len} tokens, computing for remaining {len(prompt_tuple) - prefix_len} tokens")
            
            # Here's the issue: we're processing the remaining tokens BEFORE generate
            # So by the time we call generate, past_kv already has the full prompt
            remaining_tokens = prompt_tokens[:, prefix_len:]
            
            # Forward pass for remaining tokens to extend the cache
            with torch.no_grad():
                _, _, past_kv = model(
                    remaining_tokens.to(device),
                    past_kv=past_kv,
                    return_past=True,
                )
            
            # Now past_kv contains KV for the ENTIRE prompt
            start_pos = len(prompt_tuple)  # All prompt tokens are now cached
    else:
        print("No prefix match found - computing full KV cache")
        start_pos = 0
    
    # Generate new tokens
    # If start_pos == len(prompt_tuple), generate will skip processing prompt
    # If start_pos == 0, generate will process the entire prompt
    y, final_past_kv = model.generate(
        prompt_tokens.to(device),
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_kv_cache=True,
        past_kv=past_kv,
        prefix_len=start_pos,  # Pass how many tokens are already in past_kv
    )
    
    # Cache the full prompt's final KV state
    if prompt_tuple not in prompt_kv_cache:
        prompt_kv_cache[prompt_tuple] = final_past_kv
    
    return y, final_past_kv

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
#start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
prompt_kv_cache = {} # dictionary mapping prompt strings to their precomputed KV cache tensors
prompts = [
    "Now I see the issue! The problem is that your generate function is using prefix_len incorrectly. When you have a prefix match and past_kv already contains the cached KV for the prefix, you should NOT process the prefix tokens again. But the current generate function logic doesn't handle this correctly.Let me fix the generate function to properly handle the prefix_len parameter when past_kv is provided:",
    "Now I see the issue! The problem is that your generate function is using prefix_len incorrectly. When you have a prefix match and past_kv already contains the cached KV for the prefix, you should NOT process the prefix tokens again. But the current generate function logic doesn't handle this correctly.Let me fix the generate function to properly handle the prefix_len parameter when past_kv is provided:"
]
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
use_kv_cache = True # toggle KV caching on/off for generation
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
#if start.startswith('FILE:'):
#    with open(start[5:], 'r', encoding='utf-8') as f:
#        start = f.read()
#start_ids = encode(start)
#x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
#input_tokens = x.shape[1]
#print(f"Input prompt tokens: {input_tokens}")

encoded_prompts = [
    torch.tensor(encode(p), dtype=torch.long)[None, ...]
    for p in prompts
]

# run generation
with torch.no_grad():
    with ctx:
        for i, x in enumerate(encoded_prompts):
            x = x.to(device)
            print(f"\n=== Prompt {i+1}/{len(encoded_prompts)} ===")
            print(f"Prompt: {prompts[i]}")
            print(f"Prompt length: {x.shape[1]} tokens")
            
            for k in range(num_samples):
                start_time = time.time()
                
                # Use the new prefix caching function
                y, past_kv = generate_with_prefix_caching(
                    model=model,
                    prompt_tokens=x,
                    prompt_kv_cache=prompt_kv_cache,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    device=device,
                )

                end_time = time.time()
                print(f"\n[Sample {k+1}/{num_samples}]")
                print(decode(y[0].tolist()))

                total_time = end_time - start_time
                time_per_token = total_time / max_new_tokens

                print(f"Total inference time: {total_time:.4f} seconds")
                print(f"Time per output token: {time_per_token:.6f} seconds")
                print('---------------')

