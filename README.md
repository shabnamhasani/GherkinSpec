GherkinSpec is a research-driven project that explores the use of Large Language Models (LLMs) to translate regulatory provisions into Gherkin specifications. This approach aims to bridge the gap between complex legal texts and executable specifications, facilitating Behavior-Driven Development (BDD) and enhancing compliance automation.
Gherkin specifications in our study were generated using the chat interfaces of Llama (via HuggingFace Chat: https://huggingface.co/chat/ and Claude via https://claude.ai/. Consequently, we used the default model parameters.
This repository includes:
1) Our study data under Data (prompt, unique tasks, consent form, interview protocol, recruitment form, and (regulatory provisions, Gherkin specifications, participant ratings, and feedback) under input)
2) Evaluation code under Code (boxplot, dataprocessing, wilcoxon_stats, main)
3) Evaluation results under Evaluation (plots and statistical analyses: stacked_bar, token_stats, wilcoxon)
