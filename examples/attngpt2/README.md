### Attention head explorer

Trained weights and input samples taken from [Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT) (('gpt2' config)](https://github.com/karpathy/nanoGPT/blob/master/model.py#L217)).

Sequence length has been reduced to 256 for ease of visualization, but weights retain their original dimensions.

* d_seq=256, d_emb=768, d_head=64
* 10 sample inputs
* 12 layers
* 12 attention heads

Visualizes the operation of a single attention head. 

```
input[d_seq, d_emb]
wQ[d_emb, d_head]
wK[d_emb, d_head]
wV[d_emb, d_head]
wO[d_head, d_emb]

Q = input @ wQ
K_t = wK.T @ input.T
attn = softmax(tril(Q @ K_t / sqrt(d_head)))
V = input @ wV
head_out = (attn @ V) @ wO
```

[Try it here](https://bhosmer.github.io/mm/examples/attngpt2/index.html)
