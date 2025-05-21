import os
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


def get_sentence_embedding(sentence):
    # Load pre-trained model tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
    # Get the hidden states from the model
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def get_sentence_embedding_minilm(sentences):
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    # Sentences we want sentence embeddings for
    # sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained('pritamdeka/S-PubMedBert-MS-MARCO')
    model = AutoModel.from_pretrained('pritamdeka/S-PubMedBert-MS-MARCO')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


def relative_difference(row):
    time1 = row['time1']
    time2 = row['time2']
    if time1 == 0 and time2 == 0:
        return 0  # If both are zero, the relative difference is defined as zero
    else:
        return (time1 - time2) / max(abs(time1), abs(time2), 1)

def compare_embedding(df1, df2):

    # df1 = pd.read_csv('./data/agree_annov1.csv')
    # df2 = pd.read_csv('./data/agree_annov2.csv')
    for index in range(len(id1)):# 20
        key = id1[index]#'PMC6984996'
        time1 = df1[df1['Report']==key]['Time'].values
        time2 = df2[df2['Report']==key]['Time'].values
        
        event1 = df1[df1['Report']==key]['Event'].values
        event2 = df2[df2['Report']==key]['Event'].values
        
        embeddings1 = []
        embeddings2 = []
        
        # Process all elements in event1
        for i in range(len(event1)):
            tmp1 = get_sentence_embedding(event1[i]).numpy()
            embeddings1.append(tmp1)
        
        # Process all elements in event2
        for i in range(len(event2)):
            tmp2 = get_sentence_embedding(event2[i]).numpy()
            embeddings2.append(tmp2)
        
        # Concatenate embeddings along the first axis
        emb1 = np.concatenate(embeddings1, axis=0)
        emb2 = np.concatenate(embeddings2, axis=0)
        
        distance_matrix = pairwise_distances(emb1, emb2, metric='cosine')
        
def get_and_write_embeddings(f1, f2, outfile):
    df1 = None
    
    try:
        with open(f1, "r") as f:
            # Process the file contents here
            df1 = pd.read_csv(f1, keep_default_na=False)
            # print(df1)

    except FileNotFoundError:
        print(f"Error: File '{f1}' not found.")

    df2 = None
    try:
        with open(f2, "r") as f:
            # Process the file contents here
            df2 = pd.read_csv(f2, keep_default_na=False)

    except FileNotFoundError:
        print(f"Error: File '{f2}' not found.")

    if df2 is not None and df1 is not None:
        # embs1 = torch.cat([get_sentence_embedding(e) for e in df1['event'].tolist()]).numpy()
        # embs2 = torch.cat([get_sentence_embedding(e) for e in df2['event'].tolist()]).numpy()
        # Getting better results with MiniLM sentence embedder
        # embs1 = get_sentence_embedding_minilm(df1['event'].tolist()).numpy()
        # embs2 = get_sentence_embedding_minilm(df2['event'].tolist()).numpy()
        embs1 = get_sentence_embedding_minilm([f[:512] for f in df1['event'].tolist()]).numpy()
        embs2 = get_sentence_embedding_minilm([f[:512] for f in df2['event'].tolist()]).numpy()
        # Also, the pairing is better when shorter is matched into longer; in general, we need a process for multi-match (recursive match?)
        distance_matrix = pairwise_distances(embs1, embs2, metric='cosine')
        # print(distance_matrix)
        outdf = pd.DataFrame(distance_matrix)
        # print(outdf)
        outdf.to_csv(outfile, index=False)

def main():
    parser = argparse.ArgumentParser(description="Reads a file and get embedding.")
    parser.add_argument("f1", help="The name of the file 1 to read")
    parser.add_argument("f2", help="The name of the file 2 to read")
    parser.add_argument("outfile", help="Where to save the similarity matrix")

    args = parser.parse_args()
    get_and_write_embeddings(args.f1, args.f2, args.outfile)
        
        

# if __name__ == "__main__":
#     main()