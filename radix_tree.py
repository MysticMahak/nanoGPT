import torch

class RadixTreeNode:
    def __init__(self, token_ids):
        self.token_ids = token_ids  # List of token IDs representing the path to this node
        self.children = {}  # Dictionary mapping next token ID to child node
        self.kv_cache = None  # Cached key-value pairs for this node

class RadixTree:
    def __init__(self):
        self.root = RadixTreeNode([])

    def match(self, token_ids):
        node = self.root
        matched_length = 0
        kv_segments = []

        i = 0
        while i < len(token_ids):
            tok = token_ids[i]
            if tok not in node.children:
                break

            child = node.children[tok]
            toks = child.token_ids

            j = 0
            while j < len(toks) and i < len(token_ids) and toks[j] == token_ids[i]:
                j += 1
                i += 1

            if j < len(toks):
                return node, child, j, matched_length, kv_segments
            
            kv_segments.append(child.kv_cache)
            matched_length += len(toks)
            node = child

        return node, None, 0, matched_length, kv_segments
    
    def concat_kv(self, kv_segments):
        if not kv_segments:
            return None
        
        out = []

        for layer in range(len(kv_segments[0])):
            ks, vs = [], []
            for seg in kv_segments:
                k, v = seg[layer]
                ks.append(k)
                vs.append(v)
            out.append((torch.cat(ks, dim=2), torch.cat(vs, dim=2)))

        return out
    
    