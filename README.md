# SUVI Arm Optimization

SUVI is a visually grounded, single-stream multimodal architecture. This
repository deploys and optimizes SUVI on Arm cloud infrastructure.

SUVI was introduced in the IEEE CAI 2026 paper, [SUVI: Scalable Unified Vector
Intelligence for Efficient Edge Deployment](https://ieeexplore.ieee.org/document/11536337).
This work is developed by Nakul Vyas at [Heysuvi Labs, LLC](https://heysuvi.com).

## SUVI

SUVI converts an image into a visual prefix and combines it with text in one
Mistral decoder stream. Cached images use a 1 KB FSQ-ID record, the learned
visual bridge, and the decoder.

The Phase 3 architecture passed held-out Flickr30k causal-prefix tests: real
image prefixes outperformed shuffled, fixed, and random controls.

This release is an architecture and visual-grounding checkpoint. It proves that
the visual path carries image-specific information; it is not presented as a
finished caption-quality model. More training is the next step for richer,
production-grade captions.

[Architecture](docs/ARCHITECTURE.md)

## Checkpoint

The public Phase 3 Flickr30k inference checkpoint is available on
[Hugging Face](https://huggingface.co/nakulvyas21/suvi-phase3-flickr30k-arm).
It includes the SUVI adapter, visual bridge, tokenizer files, and the frozen
Phase 3 contract. Follow the [Arm setup guide](docs/ARM64_GUIDE.md) to run it.

https://huggingface.co/nakulvyas21/suvi-phase3-flickr30k-arm

## Presentation

[SUVI: Single-Stream Vision AI presentation](presentation/SUVI_Single_Stream_Vision_AI_Presentation.pdf)

## Arm results

Measured on an 8-core Google Axion Arm VM.

| Deployment path | Peak RSS | Median request time |
| --- | ---: | ---: |
| LLaVA v1.6 Mistral-7B dual tower | 20.72 GB | 26.01 s |
| SUVI uncached | 15.36 GB | 18.79 s |
| SUVI cached | 15.36 GB | 18.12 s |

Cached SUVI used 25.9% less peak RSS and 30.3% lower median request latency
than the dual-tower comparator. Cached and uncached SUVI produced identical
visual prefixes and caption token IDs.

| Arm threads | Cached median request time |
| ---: | ---: |
| 1 | 54.98 s |
| 2 | 36.50 s |
| 4 | 24.76 s |
| 8 | 18.12 s |

[Results](docs/RESULTS.md) · [Setup](docs/ARM64_GUIDE.md)

## More from Heysuvi

Heysuvi also develops MSHI, a proprietary captioning model with demonstrated
caption quality that runs without a GPU. For partnership or product inquiries,
contact [nvyas@heysuvi.com](mailto:nvyas@heysuvi.com).

## License

Apache License 2.0. See [LICENSE](LICENSE).
