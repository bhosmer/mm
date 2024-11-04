### Attention head explorer

Visualizes the operation of a single attention head using [premultiplied QK/OV matrices](https://bhosmer.github.io/mm/examples/attnqkov/index.html):
```
head_out =
    softmax(
        tril(
            input @ (wQ @ wK.T) @ input.T
            / sqrt(d_head)
        )
    )
    @ input
    @ (wV @ wO)
```


Instead of the [more conventional formulation](https://bhosmer.github.io/mm/examples/attngpt2/index.html):
```
head_out =
    softmax(
        tril(
            (input @ wQ) @ (wK.T @ input.T)
            / sqrt(d_head)
        )
    )
    @ (input @ wV)
    @ wO
```

Trained weights and input samples taken from [Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT) (['gpt2' config](https://github.com/karpathy/nanoGPT/blob/master/model.py#L217)).

Sequence length has been reduced to 256 for ease of visualization, but weights retain their original dimensions.
```
* d_seq=256, d_emb=768, d_head=64
* 10 sample inputs
* 12 layers
* 12 attention heads
```
