# BP<sup>C</sup>: A Benchmark for Causal Business Process Reasoning with LLMs
## Abstract
Large Language Models (LLMs) have become increasingly popular tools for boosting organizational efficiency and for the automation of tasks. While not originally designed to aid with complex cognitive processes, recent integration efforts have further extended to attempts to engage LLMs in activities such as reasoning, planning, and decision-making. In business processes, such abilities could be invaluable for leveraging on the massive corpora LLMs have been trained on for gaining deep understanding of such processes.  
In this work, we plant the seeds for the development of a benchmark that is focused on assessing the ability of LLMs to reason about causal and process perspectives of business operations. We uniformly refer to this view as Causally-augmented Business Processes (BP<sup>C</sup>). 
The core of the benchmark consists of a composition of a set of BP<sup>C</sup>-related situations, a set of related questions about these situations, and a set of deductive rules employed to systematically resolve the ground truth answers to these questions. Also with the power of LLMs, the seed is then instantiated into a larger-scale set of domain-specific situations and questions.
Reasoning on BP<sup>C</sup> is of crucial importance for process interventions and process improvement. Our benchmark could be used in one of two possible modalities: testing the performance of any target LLM and training an LLM to advance its capability to reason about BP<sup>C</sup>.

Paper related resources:
- Benchmark's seed: [situations, rules, and questions](https://github.com/IBM/SAX/blob/main/NLP4BPM2024/situations-rules-questions.md).
- A series of [prompts](https://github.com/IBM/SAX/blob/main/NLP4BPM2024/prompts.md) employed.
- Domain-specific benchmark for situation \#2 is accessible [here](https://github.com/IBM/SAX/blob/main/NLP4BPM2024/domain-specific-benchmark-situation-2.xslx).
- A full-scale instantiation of the benchmark is accessible [here](https://huggingface.co/datasets/ibm/BPC).
  
## How to cite
Please consider citing [our paper](https://arxiv.org/abs/2406.05506) if you use code or ideas from this project:\
Fabiana Fournier, Lior Limonad, and Inna Skarbovsky, Towards a Benchmark for Causal Business Process Reasoning with LLMs, arXiv:2406.05506, 2024.
