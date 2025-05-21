import os
import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
import itertools
import random
from tqdm import tqdm
import warnings
from datetime import datetime
import pdb


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f'Using device: {device}')


# Initialize TensorBoard
writer = SummaryWriter()


def trim_input_text(input_text: str, max_length: int, tokens_per_word: float = 1.3) -> str:
    """
    Quickly trims the input text by estimating token length without repeated tokenization.
    
    :param input_text: The full input text with sequences separated by [SEP]
    :param max_length: The maximum allowable length of tokenized input
    :param tokens_per_word: Estimated tokens per word (default is 1.3 for BERT-like tokenizers)
    :return: The trimmed input text that fits within the max length
    """
    # Split the input text into individual sequences based on the [SEP] token
    sequences = input_text.split(' [SEP] ')
    
    if len(sequences) <= 1:
        return input_text  # Already minimal, no trimming possible
    
    # Estimate token length and trim progressively
    for start_index in range(len(sequences)):
        trimmed_sequences = sequences[start_index:]
        trimmed_text = ' [SEP] '.join(trimmed_sequences)
        
        # Add back the [CLS] token at the beginning
        if not trimmed_text.startswith('[CLS]'):
            trimmed_text = '[CLS] ' + trimmed_text
        
        # Estimate the token length using the approximation
        word_count = len(trimmed_text.split())
        estimated_token_length = int(word_count * tokens_per_word)
        
        if estimated_token_length <= max_length:
            return trimmed_text
    
    # If no trimming makes it fit, return the most trimmed version
    return trimmed_text


def preprocess_data(data, tokenizer, max_length):
    """
    Preprocess the input data by trimming the input texts to fit the max length.
    
    :param data: The original pre-batched data
    :param tokenizer: The tokenizer for length calculation
    :param max_length: The maximum allowed token length
    :return: Preprocessed data with trimmed input texts
    """
    preprocessed_data = []
    for batch in tqdm(data, desc='Preprocessing and Trimming data'):
        preprocessed_batch = []
        for item in batch:
            trimmed_text = trim_input_text(item['input_text'], max_length)
            preprocessed_batch.append({
                'input_text': trimmed_text,
                'label': item['label'],
                'timeseries_num': item['timeseries_num']
            })
        preprocessed_data.append(preprocessed_batch)
    return preprocessed_data



def preprocess_and_tokenize_data(data, tokenizer, max_length, inner_batch_size=32):
    """
    Preprocess and tokenize the input data with dynamic batching inside each case report.
    
    :param data: The original pre-batched data
    :param tokenizer: The tokenizer for text tokenization
    :param max_length: The maximum allowed token length
    :param inner_batch_size: Maximum batch size within each case report
    :return: Preprocessed data with tokenized inputs and batched per case report
    """
    preprocessed_data = []
    for case_report in tqdm(data, desc='Preprocessing and Tokenizing data'):
        preprocessed_case_report = []

        # Trim and tokenize each item in the case report
        for item in case_report:
            trimmed_text = trim_input_text(item['input_text'], max_length)
            inputs = tokenizer(
                trimmed_text,
                return_tensors='pt',
                max_length=max_length,
                padding='max_length',
                truncation=True
            )
            preprocessed_case_report.append({
                'inputs': {k: v.squeeze(0) for k, v in inputs.items()},
                'label': torch.tensor(item['label'], dtype=torch.long),
                'timeseries_num': item['timeseries_num']
            })

        # Split into inner batches to avoid memory overflow
        batched_case_report = [
            preprocessed_case_report[i:i + inner_batch_size]
            for i in range(0, len(preprocessed_case_report), inner_batch_size)
        ]

        preprocessed_data.append(batched_case_report)
    return preprocessed_data



# def calculate_concordance_index(y_true, y_pred, timeseries_nums=None):
#     if timeseries_nums is None:
#         # Global concordance calculation
#         concordant = np.sum(y_true == y_pred)
#         permissible = len(y_true)
#         return concordant / permissible if permissible > 0 else 0
#     else:
#         # Create a DataFrame for easy grouping and aggregation
#         df = pd.DataFrame({
#             'y_true': y_true,
#             'y_pred': y_pred,
#             'timeseries_num': timeseries_nums
#         })
        
#         # Calculate concordance per timeseries
#         concordance_per_ts = df.groupby('timeseries_num').apply(
#             lambda group: np.mean(group['y_true'] == group['y_pred'])
#         )

#         # Return the average concordance across all timeseries
#         return concordance_per_ts.mean()

def calculate_f1_score(y_true, y_pred, timeseries_nums=None):
    if timeseries_nums is None:
        # Global F1 score calculation using the built-in f1_score function
        return f1_score(y_true, y_pred)
    
    else:
        # Create a DataFrame for easy grouping and aggregation
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'timeseries_num': timeseries_nums
        })
        
        # Calculate F1 score per timeseries using the built-in f1_score function
        f1_per_ts = df.groupby('timeseries_num').apply(
            lambda group: f1_score(group['y_true'], group['y_pred']),
            include_groups=False
        )
        
        # Return the average F1 score across all timeseries
        return f1_per_ts.mean()
    

def calculate_best_f1_score(y_true, y_pred_probs, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)  # Default thresholds from 0.1 to 0.9

    # Vectorized thresholding: create a (num_thresholds, num_samples) binary prediction matrix
    y_pred_matrix = (y_pred_probs[None, :] >= thresholds[:, None]).astype(int)
    
    # Compute F1 scores for all thresholds efficiently
    f1_scores = np.apply_along_axis(lambda y_pred: f1_score(y_true, y_pred), axis=1, arr=y_pred_matrix)
    
    # Get the best threshold
    best_idx = np.argmax(f1_scores)
    return f1_scores[best_idx], thresholds[best_idx]


# Training Function for Time Window Classification Task
def train_time_window_model(train_data, val_data, model, epochs, lr, patience, checkpoint_path, inner_batch_size=32):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1  # Track best validation F1-score
    best_threshold = None  # Store the best threshold for test
    patience_counter = 0

    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for case_report in train_data:
            for batch in case_report:
                optimizer.zero_grad()

                inputs = {k: torch.stack([item['inputs'][k] for item in batch]).to(device) 
                          for k in batch[0]['inputs'].keys()}
                labels = torch.stack([item['label'] for item in batch]).to(device)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_train_loss = total_loss / len(train_data)

        # Evaluate on validation set
        avg_val_loss, val_f1_score, val_threshold = evaluate_model_time_window(model, val_data, mode='val')

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, "
              f"F1 Score: {val_f1_score:.4f}, Best Threshold: {val_threshold:.2f}")

        writer.add_scalar('time_window/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar('time_window/Val_Loss', avg_val_loss, epoch)
        writer.add_scalar('time_window/F1_Score', val_f1_score, epoch)

        # Check if current F1-score is the best so far
        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            best_threshold = val_threshold  # Save best threshold for test
            patience_counter = 0  # Reset patience
            torch.save({'model_state_dict': model.state_dict(), 'best_threshold': best_threshold}, checkpoint_path)
            print(f"New best model saved with F1-score: {best_val_f1:.4f} and threshold: {best_threshold:.2f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return best_val_f1, best_threshold  # Return best F1-score and threshold for testing


# Evaluation Function for Concordance Model
def evaluate_model_time_window(model, val_data, inner_batch_size=32, best_threshold=None, mode='val'):
    """
    Evaluates the model on the validation/test set.
    
    - During validation (`mode='val'`), it finds the best threshold for F1-score.
    - During test (`mode='test'`), it uses the best threshold found during validation.

    Args:
        model: The trained PyTorch model.
        val_data: Validation or test data.
        inner_batch_size: Batch size for processing cases.
        best_threshold: Threshold to use for binary classification (used in test mode).
        mode: 'val' for validation (finds best threshold), 'test' for testing (uses best threshold).
    
    Returns:
        avg_loss: Average loss over all batches.
        avg_f1_score: Average F1-score over all batches.
        best_threshold (only in validation mode).
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for case_report in val_data:
            for batch in case_report:
                inputs = {k: torch.stack([item['inputs'][k] for item in batch]).to(device) 
                          for k in batch[0]['inputs'].keys()}
                labels = torch.stack([item['label'] for item in batch]).to(device)

                outputs = model(**inputs)
                loss = nn.CrossEntropyLoss()(outputs.logits, labels)
                total_loss += loss.item()

                # Get probabilities for class 1
                preds = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                labels = labels.cpu().numpy()

                # Collect all predictions and labels for threshold selection
                all_preds.extend(preds)
                all_labels.extend(labels)

    # Convert lists to NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if mode == 'val':
        # Find the best threshold using validation data
        best_f1, best_threshold = calculate_best_f1_score(all_labels, all_preds)
        print(f'Best Threshold Found: {best_threshold:.2f}')
    else:
        # Use the provided best threshold for testing
        assert best_threshold is not None, "best_threshold must be provided during test mode."
        best_f1 = f1_score(all_labels, (all_preds >= best_threshold).astype(int))

    avg_loss = total_loss / len(val_data)
    print(f'Mode: {mode}, F1 Score: {best_f1:.4f}')
    # writer.add_scalar(f'time_window/{mode}_F1_Score', best_f1)

    if mode == 'val':
        return avg_loss, best_f1, best_threshold  # Return threshold in validation mode
    else:
        return avg_loss, best_f1  # No threshold return in test mode


# Load data from CSV files
# Function to process all CSV files and drop problematic rows
def load_data_from_directory(directory_path, verbose=False):
    all_data = []
    invalid_files = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # Check if the second column exists
            if df.shape[1] > 1:
                second_column = df.iloc[:, 1]  # Get the second column

                # Function to check if a value is a valid number (int or float)
                def is_valid_number(value):
                    if pd.isna(value):
                        return False  # Explicitly handle NaN values as invalid
                    if isinstance(value, (int, float)):
                        return True
                    if isinstance(value, str):
                        try:
                            float(value)
                            return True
                        except ValueError:
                            return False
                    return False

                # Identify invalid rows in the second column
                invalid_rows = ~second_column.apply(is_valid_number)

                if invalid_rows.any():
                    invalid_files[filename] = second_column[invalid_rows].tolist()

                    # Drop rows with invalid or NaN values in the second column
                    initial_row_count = len(df)
                    df = df[~invalid_rows].reset_index(drop=True)
                    final_row_count = len(df)

                    if verbose: print(f"Dropped {initial_row_count - final_row_count} invalid rows from {filename}.")

                # Append the cleaned DataFrame to the list
                all_data.append((filename, df))
            else:
                # Append non-problematic files as is
                all_data.append((filename, df))

    if invalid_files:
        if verbose: print("Files with non-numeric or NaN values in the second column:")
        for file, values in invalid_files.items():
            if verbose: print(f"File: {file}, Invalid Values: {values}")
    else:
        if verbose: print("All files have valid numeric values in the second column.")
    
    return [df for filename, df in all_data]


def generate_timeseries_and_events(data, K=8, timestep_drop_rate=0):
    timeseries_data_batches = []
    for df in tqdm(data):
        events = df.values.tolist()
        Ln = len(events)
        timeseries_data = []
        
        i = 0
        history = []
        
        # Step 1: Initialize history with all events at the first timestamp
        if i < Ln:
            first_time = events[i][1]
            while i < Ln and events[i][1] == first_time:
                history.append(events[i])
                i += 1
            # Generate the first timeseries with the initial set of same-timestamp events
            next_events = events[i:i+K]
            timeseries_data.append((history[:], next_events))

        # Step 2: Generate subsequent timeseries
        while i < Ln - K:
            last_time = history[-1][1]
            
            # Add all subsequent events with the same timestamp
            while i < Ln - K and events[i][1] == last_time:
                history.append(events[i])
                i += 1
            
            # Move to the next unique timestamp and add those events to history
            if i < Ln - K:
                last_time = events[i][1]
                while i < Ln - K and events[i][1] == last_time:
                    history.append(events[i])
                    i += 1
            
            # Collect the next K events
            next_events = events[i:i+K]
            timeseries_data.append((history[:], next_events))

        timeseries_data_batches.append(timeseries_data)

    # Step 3: Drop timesteps randomly after processing
    if timestep_drop_rate > 0:
        for timeseries_data in timeseries_data_batches:
            for idx, (history, next_events) in enumerate(timeseries_data):
                if len(history) > 1:  # Ensure at least one remains
                    num_to_drop = int(len(history) * timestep_drop_rate)
                    num_to_drop = min(num_to_drop, len(history) - 1)  # Keep at least 1 event
                    indices_to_drop = set(random.sample(range(len(history)), num_to_drop))

                    history = [e for i, e in enumerate(history) if i not in indices_to_drop]
                    timeseries_data[idx] = (history, next_events)  # Update in-place
    
    return timeseries_data_batches


# Prepare dataset for the two tasks
def prepare_dataset(timeseries_data, K=8, H=24, tokenizer=None, max_length=512, inner_batch_size=32):
    concordance_data_batches = []
    time_classification_data_batches = []

    for batch_index, batch in enumerate(timeseries_data):
        concordance_data = []
        time_classification_data = []

        for timeseries_num, (history, next_events) in enumerate(batch):

            # Prepare time classification data
            last_time = float(history[-1][1])
            for event, time in next_events:
                label = 1 if (float(time) - last_time) <= H else 0
                input_text = f"[CLS] {' [SEP] '.join([f'({e[1]}) : '+e[0] for e in history])} [SEP] {event} [SEP]"

                # Trim before tokenizing
                trimmed_text = trim_input_text(input_text, max_length)

                # Tokenize trimmed input
                tokenized_input = tokenizer(trimmed_text, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True)
                #label_token_id = tokenizer.convert_tokens_to_ids(label)
                #torch.tensor(item['label'], dtype=torch.long

                time_classification_data.append({
                    'inputs': {k: v.squeeze(0) for k, v in tokenized_input.items()},
                    'label': torch.tensor(label, dtype=torch.long),
                    'timeseries_num': timeseries_num})

        
        if len(time_classification_data) > 0:

            # Batch inner data like before
            batched_case_report = [
                time_classification_data[i:i + inner_batch_size]
                for i in range(0, len(time_classification_data), inner_batch_size)]
            
            time_classification_data_batches.append(batched_case_report)

    return time_classification_data_batches



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=8, help='Number of next events to consider')
    parser.add_argument('--H', type=int, default=24, help='Time window for classification')
    parser.add_argument('--cache_dir', type=str, default="/data/kumars33/.cache/huggingface/hub/", help='Cache directory')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base', help='Pretrained model name')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum token length for the model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--checkpoint_path_time_window', type=str, default='best_model_time_window.pth', help='Checkpoint path for time window model')
    parser.add_argument('--train_data_directory', type=str, default='/data/CHARM-MIMIC/kumars33/tts_forecast/tts_files/L33_annotations/ordered_new_l33train', help='Directory containing training data')
    parser.add_argument('--test_data_directory',  type=str, default='/data/CHARM-MIMIC/kumars33/tts_forecast/tts_files/L33_annotations/ordered_new_l33test', help='Directory containing Test data')
    parser.add_argument('--test_results_text_file', type=str, default='test_F1_indices.txt', help='File to save test results')
    parser.add_argument('--timestep_drop_rate', type=float, default=0, help='Drop rate for timesteps')
    parser.add_argument('--only_eval', action='store_true', default=False, help='Enable only evaluation from checkpoint')

    args = parser.parse_args()

    K = args.K
    H = args.H

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir = args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2, cache_dir = args.cache_dir).to(device)

    try:
        max_model_context = model.config.max_position_embeddings
        if max_model_context < args.max_length:
            args.max_length = max_model_context
            print(f"Max model length is changed {args.max_length} --> Max Model Context Length: {max_model_context}")
    except Exception as e:
        warnings.warn(f"Couldn't retrieve max model context for {args.model_name}. Make sure --max_length is set appropriately", RuntimeWarning)


    if not args.only_eval:

        # Load and preprocess training data

        train_data = load_data_from_directory(args.train_data_directory)
        train_timeseries_data = generate_timeseries_and_events(train_data, K, timestep_drop_rate=args.timestep_drop_rate)
        time_classification_data = prepare_dataset(train_timeseries_data, tokenizer=tokenizer, max_length=args.max_length, inner_batch_size=args.batch_size, H=H)
        time_classification_data_train, time_classification_data_val = train_test_split(time_classification_data, test_size=0.2, shuffle=True)

        # Train the model and find the best threshold
        print("Training Time Window Task")
        best_f1, best_threshold = train_time_window_model(
            time_classification_data_train, 
            time_classification_data_val, 
            model, 
            epochs=args.epochs, 
            lr=args.lr, 
            patience=args.patience, 
            checkpoint_path=args.checkpoint_path_time_window,
            inner_batch_size=args.batch_size
        )

    # Load the best model and its associated threshold for evaluation
    print("Evaluating on Test Set")
    checkpoint = torch.load(args.checkpoint_path_time_window, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_threshold = checkpoint['best_threshold']

    # Load and preprocess test data
    test_data = load_data_from_directory(args.test_data_directory)
    test_timeseries_data = generate_timeseries_and_events(test_data, K, timestep_drop_rate=args.timestep_drop_rate)
    time_classification_data_test = prepare_dataset(test_timeseries_data, tokenizer=tokenizer, max_length=args.max_length, inner_batch_size=args.batch_size, H=H)

    # Evaluate on test set using best threshold
    _, f1_score_test = evaluate_model_time_window(
        model, time_classification_data_test, best_threshold=best_threshold, mode='test'
    )

    print(f" Model name: {args.model_name} Test F1-Score: {f1_score_test:.4f} (Using Best Threshold: {best_threshold:.2f})")

    # Save the test F1 score and threshold to a file with a timestamp
    with open(args.test_results_text_file, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} | Model: {args.model_name} | Test F1: {f1_score_test:.4f} | Best Threshold: {best_threshold:.2f}\n")