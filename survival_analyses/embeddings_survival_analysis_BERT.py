import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="<Insert data directory>")  # e.g., phe_deceased/sepsis10
    parser.add_argument('--cache_dir', type=str, default="<Insert cache directory>", help='Cache directory')
    parser.add_argument('--model_name', type=str, default='answerdotai/ModernBERT-base')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--output_pickle', type=str, default=None)
    parser.add_argument('--time_of_interest', type=int, default=0)  # Default to 0 if not provided


    # Set the output_pickle default dynamically based on model_name if not provided
    args = parser.parse_args()
    embeddings_folder = os.path.join(args.data_folder, 'llm_embeddings_without_time')

    if args.output_pickle is None:
        args.output_pickle = f"{embeddings_folder}/survival_data_embeddings_{args.model_name.split('/')[-1]}_t{args.time_of_interest}.pkl"

    return args

# def get_label_info(clean_file_path):
#     line = open(clean_file_path).readline().strip()
#     label_str, time = line.split('|')
#     event = 0 if label_str.strip() == 'censored' else 1
#     try:
#         time = float(time.strip())
#     except ValueError:
#         print("File path:", clean_file_path)
#         print("Line content:", line)
#         time = None
#     return time, event


def get_label_info(clean_file_path):
    try:
        line = open(clean_file_path).readline().strip()
    except Exception as e:
        print(f"Error reading file: {clean_file_path} — {e}")
        return None, None

    parts = line.split('|')
    if len(parts) != 2:
        print(f"Skipping malformed file: {clean_file_path}")
        print(f"Line content: '{line}'")
        return None, None  # Indicate invalid file format
    
    label_str, time = parts
    event = 0 if label_str.strip().lower() == 'censored' else 1
    
    try:
        time = float(time.strip())
    except ValueError:
        print("Invalid time format.")
        print("File path:", clean_file_path)
        print("Line content:", line)
        return None, None
    
    return time, event

# def get_event_history(csv_file_path, time_of_interest=0):
#     df = pd.read_csv(csv_file_path)

#     # Define number check
#     def is_valid_number(value):
#         if pd.isna(value):
#             return False
#         if isinstance(value, (int, float)):
#             return True
#         if isinstance(value, str):
#             try:
#                 float(value)
#                 return True
#             except ValueError:
#                 return False
#         return False

#     # Identify invalid rows
#     invalid_rows = ~df['time'].apply(is_valid_number)
#     df = df[~invalid_rows].copy()

#     df['time'] = df['time'].astype(float)  # ensure numeric type

#     # Sort by time ascending
#     df = df.sort_values(by='time', ascending=True)

#     # Select only events with time <= time
#     history = [(row['event'], row['time']) for _, row in df.iterrows() if row['time'] <= time_of_interest]

#     return " [SEP] ".join([f"{e} ({t}h)" for e, t in history])


def get_event_history(csv_file_path, time_of_interest=0):

    #print(csv_file_path)

    df = pd.read_csv(csv_file_path)

    # Skip empty files with just column headers (no rows)
    if df.empty:
        print(f"Skipping empty file: {csv_file_path}")
        return

    # Define number check
    def is_valid_number(value):
        if pd.isna(value):
            return False
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False
        return False

    if 'time' not in df.columns:
        print(csv_file_path)
        raise ValueError(f"'time' column not found")

    # Identify invalid rows
    invalid_rows = ~df['time'].apply(is_valid_number)
    df = df[~invalid_rows].copy()

    df['time'] = df['time'].astype(float)  # ensure numeric type

    # Sort by time ascending
    df = df.sort_values(by='time', ascending=True)

    # Select only events with time <= 0
    # history = [(row['event'], row['time']) for _, row in df.iterrows() if row['time'] <= time_of_interest]
    # return " [SEP] ".join([f"{e} ({t}h)" for e, t in history])

    history = [(row['event']) for _, row in df.iterrows() if row['time'] <= time_of_interest]
    return " [SEP] ".join([f"{e}" for e in history])





def compute_embeddings(texts, model, tokenizer, max_length, device):

    # Make sure texts is a list of strings
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, list):
        texts = [str(t) for t in texts]  # Ensures all elements are strings
    else:
        raise TypeError("Expected texts to be a string or a list of strings.")

    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
    return cls_embeddings.cpu()


def main():
    args = parse_args()
    tts_folder_path = os.path.join(args.data_folder, 'sampled_25k_timeord')  ## folder with time-ordered CSV files with textual time-series
    event_time_folder_path = os.path.join(args.data_folder, 'temp_llm_death_phe_output/clean') # folder with death phenotype labels. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir = args.cache_dir)
    model = AutoModel.from_pretrained(args.model_name, cache_dir = args.cache_dir).to(device)
    model.eval()

    all_filenames = [f for f in os.listdir(event_time_folder_path) if f.endswith('.bsv')]
    batch_texts, batch_filenames, batch_labels = [], [], []
    results = {}

    for filename in tqdm(all_filenames):
        bsv_path = os.path.join(event_time_folder_path, filename)
        csv_path = os.path.join(tts_folder_path, filename.replace('.bsv', '.csv'))

        if not os.path.exists(csv_path):
            continue

        time, event = get_label_info(bsv_path)
        if time is None:
            print(f"Skipping {filename}: invalid time")
            continue
        time = time - args.time_of_interest  # Adjust time based on the time of interest (time until event from the last time point)
        # print(f"Processing {filename}: time={time}, event={event}")
        text = get_event_history(csv_path, args.time_of_interest)

        batch_texts.append(text)
        batch_filenames.append(filename.replace('.bsv', ''))
        batch_labels.append((time, event))  # Save the label (time, event) for this batch item

        if len(batch_texts) >= args.batch_size:
            embeddings = compute_embeddings(batch_texts, model, tokenizer, args.max_length, device)
            for name, emb, label in zip(batch_filenames, embeddings, batch_labels):
                results[name] = {
                    'embedding': emb,
                    'label': label  # Now using the correct label for each filename
                }
            batch_texts, batch_filenames, batch_labels = [], [], []  # Clear the batch

    # process remaining
    if batch_texts:
        embeddings = compute_embeddings(batch_texts, model, tokenizer, args.max_length, device)
        for name, emb, label in zip(batch_filenames, embeddings, batch_labels):
            results[name] = {
                'embedding': emb,
                'label': label  # Now using the correct label for each filename
            }

    with open(args.output_pickle, 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
