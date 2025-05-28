This folder contains scripts, utilities, and notebooks to perform survival analysis on embeddings derived from clinical text representations. The folder termed "survival_analysis" evaluates various survival models across different experimental subsets using deep learning-based survival models.

# Survival experiments overview

We include a classical survival analysis task to model time until death. 
Many case reports specify whether the patient died and, if so, when (e.g., “the patient died on hospital day 10”), which enables us to define a time-to-event outcome. 
For this task, we evaluate models at predefined cut-off times -- specifically at 0 hours (admission), 24 hours (1 day), and 168 hours (1 week) -- and use the extracted event history up to each cut-off as input. 
A survival model is trained to predict the probability of survival over time beyond each cut-off. 
We evaluate performance using the time-dependent concordance index to assess how well the predicted survival times align with actual outcomes.

To evaluate the prognostic value of textual information encoded in large language models (LLMs), we adopt a two-stage framework: 
- (1) extraction of fixed-dimensional sequence embeddings from various pre-trained LLMs
- (2) downstream survival modeling using these embeddings as covariates.

## Embedding Extraction

In the first stage, we process each textual time series (which is converted to a textual context of \emph{"(time) clinical event [SEP] ... [SEP] (time) clinical event [SEP]"}) using a suite of LLMs to obtain dense vector representations that summarize the content of the input sequence. 
For models belonging to the encoder family, including *bert-base-uncased*, *roberta-base*, *deberta-v3-small*, *ModernBERT-base*, and *ModernBERT-large*, we extract the final hidden state corresponding to the [CLS] token, which is conventionally used to represent the entire sequence in classification tasks. 
This token-specific embedding serves as a compact, sequence-level representation.

For decoder-based models that do not utilize a dedicated [CLS] token, including *DeepSeek-R1-Distill-Llama-70B*, *Llama-3.3-70B-Instruct*, *DeepSeek-R1-Distill-Llama-8B*, and *Llama-3.1-8B-Instruct*, we compute the mean-pooled embedding over the last hidden states of all non-padding tokens. 
This pooling strategy yields a fixed-length vector that captures the overall semantic content of the input while mitigating the impact of padding artifacts.

In the second stage, these embedding vectors are used as input covariates to three survival models (see subfolder named "survival_analysis").

## 📁 Folder Structure
```
📄 embeddings_survival_analysis_BERT.py     # python script to extract embeddings for encoder-based families
📄 embeddings_survival_analysis_llm.py      # python script to extract embeddings for decoder-based families
📄 LLM_death_phenotypes_25k.py              # python script use LLMs (DeepSeek R1) to extract the death phenotype labels for a sample of 25k patients
```
