# Forecasting results

Code for reproducing the forecasting results (BERT-based models)

- L33_encoder_concordance:
- L33_encoder_mask_concordance:
- L33_encoder_mask_time_window:
- L33_encoder_time_window:

## Modeling approaches
We also evaluate a range of encoder-based models trained end-to-end on each task. 
Architectures include BERT-base-uncased, RoBERTa-base, DeBERTa-v3-small, ModernBERT-small, and ModernBERT-large. 
For each model, we append a task-specific MLP head for event occurrence or ordering prediction. 
The model input is the same flattened prefix of the event sequence used in other settings, tokenized and formatted according to the respective architecture. 
These models are trained using standard supervised learning objectives and evaluated on the same metrics as other methods: F1 score for event occurrence, and pairwise concordance for temporal ordering.


## Task definitions

Event Occurrence Prediction: Given a prefix of the clinical timeline (all events up to a certain time t), the model is tasked with predicting whether each of the immediate next k events occurs within a specified time horizon. This setup is repeated across multiple time cut-offs to simulate an “online” forecasting scenario, where the model must output a binary label for each of the next k events: does this event occur within h hours after t? Time horizons used include 1 hour, 24 hours (1 day), and 168 hours (1 week). The task is framed as a series of binary classification problems, with one binary decision per event. Evaluation is based on precision, recall, and F1 score, averaged across event positions for each time horizon.

Temporal Ordering Prediction: This task assesses the model’s ability to reconstruct the correct sequence of future events. At each time cut-off t, we extract the next k events from the timeline and remove their timestamps. The model must correctly output a permutation over these events that matches their true chronological order. This is framed as a ranking task, evaluated by computing the pairwise concordance between the predicted and true orderings (e.g., proportion of correctly ordered pairs). This tests whether the model can infer temporal progression from unordered event content alone.
