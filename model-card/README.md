---
license: apache-2.0
base_model: mistralai/Mistral-7B-v0.1
tags:
  - suvi
  - vision-language
  - image-captioning
  - arm64
---

# SUVI Phase 3 Flickr30k Arm Checkpoint

This is the public inference checkpoint for the SUVI Phase 3 visual bridge:
a visually grounded, single-stream multimodal architecture developed by Nakul
Vyas at [Heysuvi Labs, LLC](https://heysuvi.com/).

It contains the LoRA adapter, learned visual bridge, tokenizer files, and the
frozen Phase 3 contract used by the Arm optimization repository.

## Run it

Use the implementation and setup instructions in the
[SUVI Arm Optimization repository](https://github.com/nakulvyas21/suvi-arm-optimization).
The checkpoint uses `mistralai/Mistral-7B-v0.1` as its base decoder and
`stabilityai/sd-vae-ft-mse` for image encoding.

## Training and evaluation

The visual bridge was trained on Flickr30k captions. The Phase 3 architecture
passed held-out Flickr30k causal-prefix tests: real image prefixes outperformed
shuffled, fixed, and random controls.

This checkpoint is released to reproduce the SUVI visual-grounding and Arm
deployment work. Its purpose is to establish that the single-stream visual path
carries image-specific information. It is not positioned as a finished
caption-quality model; more training is the next step for richer captions.

Flickr30k images and captions are not included in this repository.

## Citation

```bibtex
@article{vyas2026suvi,
  title={SUVI: Scalable Unified Vector Intelligence for Efficient Edge Deployment},
  author={Vyas, Nakul},
  journal={IEEE Conference on Artificial Intelligence},
  year={2026},
  url={https://ieeexplore.ieee.org/document/11536337}
}
```

For partnership or product inquiries, contact
[nvyas@heysuvi.com](mailto:nvyas@heysuvi.com).
