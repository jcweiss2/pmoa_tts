# Forecasting results

Code for reproducing the forecasting results (BERT-based models)

- L33_encoder_concordance:
- L33_encoder_mask_concordance:
- L33_encoder_mask_time_window:
- L33_encoder_time_window:

We also evaluate a range of encoder-based models trained end-to-end on each task. 
Architectures include \texttt{BERT-base-uncased}, \texttt{RoBERTa-base}, \texttt{DeBERTa-v3-small}, \texttt{ModernBERT-small}, and \texttt{ModernBERT-large}. 
For each model, we append a task-specific MLP head for event occurrence or ordering prediction. 
The model input is the same flattened prefix of the event sequence used in other settings, tokenized and formatted according to the respective architecture. 
These models are trained using standard supervised learning objectives and evaluated on the same metrics as other methods: F1 score for event occurrence, and pairwise concordance for temporal ordering.
