# SUVI short presentation notes

Target length: approximately three minutes.

## Slide 1, 15 seconds

"Hi, I am Nakul Vyas from Heysuvi Labs. This is SUVI, a single-stream vision-language architecture designed for efficient multimodal AI deployment."

## Slide 2, 25 seconds

"Most vision-language systems use two large components: a language model plus a separate vision tower. That creates a large active inference path, with added memory, latency, and deployment complexity. This matters when we want multimodal AI on efficient cloud and edge systems."

## Slide 3, 30 seconds

"SUVI takes a different approach. It converts an image into discrete visual IDs, maps them through a learned bridge, and combines them with text in one Mistral decoder stream. The architecture is called Scalable Unified Vector Intelligence. The goal is not to add another tower, but to give visual evidence a native place in the token stream."

## Slide 4, 25 seconds

"Before optimizing for Arm, we established that the visual path is meaningful. On held-out Flickr30k causal-prefix tests, real image prefixes outperformed shuffled, fixed, and random controls. This shows that the architecture carries image-specific visual information. Next, we scale training for richer caption quality."

## Slide 5, 25 seconds

"For the Arm deployment case study, we added two practical optimizations. First, an image can be kept as a one-kilobyte FSQ visual record. For a cached image, it reconstructs the same visual prefix and caption token output. Second, the workload scales from one to eight Arm cores, reducing cached request time from about 55 seconds to 18 seconds."

## Slide 6, 30 seconds

"On an eight-core Google Axion Arm VM, cached SUVI used 15.36 gigabytes of peak RSS and took 18.12 seconds per median request. The LLaVA v1.6 Mistral-7B dual-tower comparator used 20.72 gigabytes and took 26.01 seconds. That is 25.9 percent less memory and 30.3 percent lower latency for cached SUVI."

## Slide 7, 20 seconds

"The result is a public, reproducible path for grounded multimodal AI with a smaller active inference footprint. Arm is one measured deployment case; the architecture is designed for a broader set of efficient cloud and edge systems. The code and checkpoint are public. Thank you."
