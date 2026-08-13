# Single-stream SUVI deployment on Arm64

SUVI's Phase 3 decoder path is single-stream: the decoder receives one
sequence of visual-prefix embeddings followed by the text prompt. It does not
run a CLIP/ViT tower or a cross-attention projector in the request path.

```mermaid
flowchart LR
  image[Permitted image] --> vae[Frozen float32 SD-VAE]
  vae --> fsq[Pool + channel-wise FSQ]
  fsq --> ids[256 x 4 integer IDs]
  ids --> cache[(Raw cache: 1,024 bytes/image)]
  ids --> bridge[Saved factorized visual bridge]
  cache --> bridge
  bridge --> prefix[256 visual-prefix embeddings]
  prompt[Text prompt] --> mistral[Mistral decoder]
  prefix --> mistral
  mistral --> caption[Caption]
```

## Request paths

| Path | Online components | Visual representation |
| --- | --- | --- |
| Uncached | image preprocessing, SD-VAE, pooling, FSQ, bridge, Mistral decoder | freshly derived FSQ IDs |
| Cached | raw ID read, bridge, Mistral decoder | the same validated FSQ IDs |

The cache stores `256 × 4` uint8 IDs, or 1,024 bytes per image. Cached and
uncached requests produce identical visual prefixes and caption token IDs.
