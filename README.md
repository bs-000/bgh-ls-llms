# LLMs \& Guiding Principles

This repository contains the code for the paper *Generating Guiding Principles:
Evaluating Large Language Models for Complex German Legal Summaries* for JURISIN 2025.

Using the task of generating guiding principles for judgments
of the German Federal Court of Justice we investigate whether current
state of the art large language models can solve a complex legal sum-
marisation task.Our results indicate that prompt engineering is not yet
sufficient to solve the task, but fine-tuning already shows promising re-
sults. In addition, our results show that models with an increased context
window size do not necessarily take the entire input into account equally.

This repository also includes our implementation of ROUGE (custom_rouge.py) which allows for a better preprocessing of German legal texts. Tests for the implementation can be found in test/test_rouge.py.