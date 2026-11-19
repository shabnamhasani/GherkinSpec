# From Law to Gherkin: A Human-Centred Quasi-Experiment on the Quality of LLM-Generated Behavioural Specifications from Food-Safety Regulations

Information and Software Technology, volume 195, article 108122, 2026. [Paper (DOI)](https://doi.org/10.1016/j.infsof.2026.108122) · [arXiv](https://arxiv.org/abs/2508.20744)

GherkinSpec is a research-driven project that explores the use of Large Language Models (LLMs) to translate regulatory provisions into Gherkin specifications. This approach aims to bridge the gap between complex legal texts and executable specifications, facilitating Behavior-Driven Development (BDD) and enhancing compliance automation.
Gherkin specifications in our study were generated using the chat interfaces of Llama (via HuggingFace Chat: https://huggingface.co/chat/ and Claude via https://claude.ai/. Consequently, we used the default model parameters.
This repository includes:
1) Our study data under Data (prompt, unique tasks, consent form, interview protocol, recruitment form, and (regulatory provisions, Gherkin specifications, participant ratings, and feedback) under input)
2) Evaluation code under Code (boxplot, dataprocessing, wilcoxon_stats, main)
3) Evaluation results under Evaluation (plots, statistical analyses, and qualitative findings: stacked_bar, token_stats, wilcoxon, qualitative results)

## Citation

```bibtex
@article{Hassani_2026,
  title    = {From law to Gherkin: A human-centred quasi-experiment on the quality of LLM-generated behavioral specifications from food-safety regulations},
  author   = {Hassani, Shabnam and Sabetzadeh, Mehrdad and Amyot, Daniel},
  journal  = {Information and Software Technology},
  volume   = {195},
  pages    = {108122},
  year     = {2026},
  month    = {July},
  doi      = {10.1016/j.infsof.2026.108122}
}
```











