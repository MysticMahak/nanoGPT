"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

@dataclass
class RadixTreeNode:
    children: Dict[str, 'RadixTreeNode']
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    token_ids: Optional[List[int]] = None
    computed: bool = False  # Whether KV cache has been computed

class RadixTree:
    def __init__(self):
        self.root = RadixTreeNode(children={})
    
    def build_from_batch(self, batch_prompts: List[List[int]]):
        """Build radix tree from batch of prompts"""
        for tokens in batch_prompts:
            node = self.root
            i = 0
            
            while i < len(tokens):
                token = tokens[i]
                child_key = chr(token)
                
                if child_key not in node.children:
                    # Create new node for this token sequence
                    new_node = RadixTreeNode(
                        children={},
                        token_ids=tokens[i:],
                        computed=False
                    )
                    node.children[child_key] = new_node
                    break
                else:
                    # Find common prefix with existing child
                    child_node = node.children[child_key]
                    child_tokens = child_node.token_ids
                    
                    # Find matching prefix length
                    j = 0
                    while (i + j < len(tokens) and 
                           j < len(child_tokens) and 
                           tokens[i + j] == child_tokens[j]):
                        j += 1
                    
                    if j == len(child_tokens):
                        # Full match, move to child
                        node = child_node
                        i += j
                    else:
                        # Partial match, need to split
                        prefix_tokens = child_tokens[:j]
                        suffix_tokens = child_tokens[j:]
                        new_branch_tokens = tokens[i + j:]
                        
                        # Create intermediate node for shared prefix
                        intermediate_node = RadixTreeNode(
                            children={},
                            token_ids=prefix_tokens,
                            computed=False
                        )
                        
                        # Update existing child to be suffix branch
                        child_node.token_ids = suffix_tokens
                        
                        # Create new branch for the current prompt
                        new_branch_node = RadixTreeNode(
                            children={},
                            token_ids=new_branch_tokens,
                            computed=False
                        )
                        
                        # Reorganize tree
                        del node.children[child_key]
                        node.children[chr(prefix_tokens[0])] = intermediate_node
                        intermediate_node.children[chr(suffix_tokens[0])] = child_node
                        intermediate_node.children[chr(new_branch_tokens[0])] = new_branch_node
                        
                        break
    
    def compute_kv_caches(self, model, device):
        """Compute KV caches for all nodes in the tree"""
        def compute_node_kv(node):
            if node.token_ids and not node.computed:
                # Convert tokens to tensor
                tokens_tensor = torch.tensor(node.token_ids, dtype=torch.long, device=device).unsqueeze(0)
                
                # Compute KV cache
                with torch.no_grad():
                    _, _, kv_cache = model(tokens_tensor, return_past=True)
                
                # Stack KV caches from all layers
                all_keys = []
                all_values = []
                for layer_idx, (key, value) in enumerate(kv_cache):
                    all_keys.append(key.unsqueeze(0))
                    all_values.append(value.unsqueeze(0))
                
                key_cache = torch.cat(all_keys, dim=0)
                value_cache = torch.cat(all_values, dim=0)
                
                node.kv_cache = (key_cache, value_cache)
                node.computed = True
                
                # Recursively compute for children
                for child in node.children.values():
                    compute_node_kv(child)
        
        compute_node_kv(self.root)
    
    def get_longest_match(self, tokens: List[int]):
        """Get the longest matching prefix and its KV cache"""
        node = self.root
        matched_tokens = []
        matched_kv = None
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            child_key = chr(token)
            
            if child_key not in node.children:
                break
                
            child_node = node.children[child_key]
            child_tokens = child_node.token_ids
            
            # Check if child sequence matches
            if (i + len(child_tokens) <= len(tokens) and 
                tokens[i:i+len(child_tokens)] == child_tokens):
                
                # Accumulate matched tokens
                matched_tokens.extend(child_tokens)
                
                # Get KV cache if this node has been computed
                if child_node.computed:
                    matched_kv = child_node.kv_cache
                
                i += len(child_tokens)
                node = child_node
            else:
                break
        
        return matched_tokens, matched_kv

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = False
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, past_kv=None):
        """
        x: (B, T, C)
        past_kv: None or tuple (past_key, past_value) where each is (B, nh, T_past, hs)

        Returns: y (B, T, C), present_kv (key, value) where key/value include past appended with new
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # If past_kv is provided, append past keys/values for caching
        if past_kv is not None:
            # past_kv is expected as a tuple (past_k, past_v)
            past_k, past_v = past_kv
            # concatenate on the time dimension
            # past_k/past_v shapes: (B, nh, T_past, hs)
            k = torch.cat([past_k, k], dim=2) if past_k is not None else k
            v = torch.cat([past_v, v], dim=2) if past_v is not None else v

        # present_kv to return for caching (note: we return the full stacked k/v)
        present = (k, v)

        # causal self-attention; Self-attend: (B, nh, T_q, hs) x (B, nh, hs, T_k) -> (B, nh, T_q, T_k)
        if self.flash:
            # when using flash attention the API supports is_causal only when T_q==T_k or so; skip for now
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # mask shape should be (1,1,T_q,T_k)
            T_q = q.size(2)
            T_k = k.size(2)
            past_len = T_k - T_q
            mask = self.bias[:, :, past_len:past_len + T_q, :T_k]
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T_q, T_k) x (B, nh, T_k, hs) -> (B, nh, T_q, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, present

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None):
        # pass past_kv for this block to attention, receive updated kv cache
        attn_out, present = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kv=None, return_past=False):
        """
        If `past_kv` is provided it should be a list of length `n_layer` where each element is
        either None or a tuple `(past_k, past_v)` with shapes `(B, nh, T_past, hs)`.

        If `return_past` is True, the function returns `(logits, loss, presents)` where `presents`
        is a list of per-layer `(k,v)` tuples (the updated cached keys/values). Otherwise returns
        `(logits, loss)` to preserve backward compatibility.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # determine position indices taking into account past kv length
        if past_kv is None:
            past_len = 0
        else:
            # infer past length from first layer's past_k if present, else zero
            first = past_kv[0]
            if first is None:
                past_len = 0
            else:
                past_len = first[0].size(2)

        pos = torch.arange(past_len, past_len + t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        presents = []
        for i, block in enumerate(self.transformer.h):
            layer_past = None if past_kv is None else past_kv[i]
            x, present = block(x, past_kv=layer_past)
            presents.append(present)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if return_past:
            return logits, loss, presents
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_kv_cache=True):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if not use_kv_cache:
            # simple generation loop without KV cache: re-forward full context each step
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

        # KV-cache enabled generation (default path)
        past_kv = None
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # For the very first step we may pass the full context so that the model builds initial cache.
            if past_kv is None:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _, past_kv = self(idx_cond, past_kv=None, return_past=True)
            else:
                # only pass the last token (the newest one) and reuse cached K/V
                idx_cond = idx[:, -1:]
                logits, _, past_kv = self(idx_cond, past_kv=past_kv, return_past=True)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_batch_with_radix(self, batch_prompts, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate completions for a batch of prompts using radix tree for KV cache sharing.
        
        Args:
            batch_prompts: List of tensors, each shape (1, seq_len)
            max_new_tokens: Number of tokens to generate for each prompt
            temperature, top_k: Sampling parameters
        
        Returns:
            List of tensors with generated sequences, each shape (1, seq_len + max_new_tokens)
        """
        import time
        device = next(self.parameters()).device
        results = []
        
        # Step 1: Extract token lists from batch
        prompt_token_lists = []
        for prompt_tensor in batch_prompts:
            prompt_token_lists.append(prompt_tensor[0].tolist())
        
        print(f"Building radix tree for {len(batch_prompts)} prompts...")
        start_tree = time.time()
        
        # Step 2: Build radix tree from all prompts
        radix_tree = RadixTree()
        radix_tree.build_from_batch(prompt_token_lists)
        
        print(f"Computing KV caches for shared prefixes...")
        start_kv = time.time()
        
        # Step 3: Compute KV caches for shared prefixes
        radix_tree.compute_kv_caches(self, device)
        
        kv_time = time.time() - start_kv
        print(f"KV cache computation time: {kv_time:.2f}s")
        
        # Step 4: Process each request sequentially
        print(f"\nGenerating outputs for {len(batch_prompts)} prompts...")
        generation_times = []
        
        for prompt_idx, (prompt_tensor, original_tokens) in enumerate(zip(batch_prompts, prompt_token_lists)):
            prompt_start = time.time()
            print(f"\nProcessing prompt {prompt_idx + 1}/{len(batch_prompts)}")
            print(f"Prompt tokens: {len(original_tokens)}")
            
            # Step 4a: Find longest matching prefix in radix tree
            matched_tokens, matched_kv = radix_tree.get_longest_match(original_tokens)
            matched_len = len(matched_tokens)
            
            if matched_len > 0:
                print(f"Matched prefix length: {matched_len} tokens")
            
            # Initialize with matched KV cache if available
            past_kv = None
            if matched_kv is not None and matched_len > 0:
                cached_key, cached_value = matched_kv
                past_kv = []
                for layer_idx in range(self.config.n_layer):
                    layer_key = cached_key[layer_idx]
                    layer_value = cached_value[layer_idx]
                    past_kv.append((layer_key, layer_value))
            
            # Step 4b: Start with original tokens
            current_tokens = original_tokens.copy()
            
            # Step 4c: Process remaining part of the prompt (if not fully matched)
            if matched_len < len(original_tokens):
                remaining_tokens = original_tokens[matched_len:]
                remaining_tensor = torch.tensor(remaining_tokens, dtype=torch.long, device=device).unsqueeze(0)
                
                if past_kv is None:
                    # No matched prefix, process from scratch
                    logits, _, past_kv = self(remaining_tensor, return_past=True)
                else:
                    # Continue from matched prefix
                    logits, _, past_kv = self(remaining_tensor, past_kv=past_kv, return_past=True)
            
            # Step 4d: Generate new tokens
            print(f"Generating {max_new_tokens} new tokens...")
            for token_idx in range(max_new_tokens):
                # Use last token as input
                last_token = torch.tensor([[current_tokens[-1]]], dtype=torch.long, device=device)
                logits, _, past_kv = self(last_token, past_kv=past_kv, return_past=True)
                
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                current_tokens.append(next_token)
                
                # Print progress for long generations
                if max_new_tokens > 50 and (token_idx + 1) % 10 == 0:
                    print(f"  Generated {token_idx + 1}/{max_new_tokens} tokens")
            
            # Store result
            results.append(torch.tensor([current_tokens], device=device))
            
            # Calculate and store generation time
            prompt_time = time.time() - prompt_start
            generation_times.append(prompt_time)
            print(f"Prompt {prompt_idx + 1} completed in {prompt_time:.2f}s")
            print(f"Total tokens: {len(current_tokens)}")
        
        # Print summary
        if generation_times:
            avg_time = sum(generation_times) / len(generation_times)
            print(f"\n=== Generation Summary ===")
            print(f"Total prompts: {len(batch_prompts)}")
            print(f"Average time per prompt: {avg_time:.2f}s")
            print(f"Total generation time: {sum(generation_times):.2f}s")
        
        total_tree_time = time.time() - start_tree
        print(f"Total radix tree processing time: {total_tree_time:.2f}s")
        
        return results

    # Optional: Add this for backward compatibility
    @torch.no_grad()
    def generate_with_radix(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Single prompt version that uses the same radix tree logic
        """
        batch_result = self.generate_batch_with_radix(
            [idx], 
            max_new_tokens, 
            temperature, 
            top_k
        )
        return batch_result[0]
