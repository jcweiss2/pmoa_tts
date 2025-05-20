from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import gzip
import random
import argparse
import time

def run(args):
    model_name = args.model_name
    cache_dir = args.cache_dir
    subdir = args.subdir
    datadir = args.datadir
    device = 'cuda'

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir=cache_dir,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map='auto')
    # The model's device_map appears to take priority over the pipeline device_map!
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=cache_dir)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        eos_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer,
        device_map="auto",
    )

    import os
    import pandas as pd

    prefix31 = r'''
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a physician. Determine the number of case reports in the following manuscript. Return 0 if it is not a case report. Reply only with the number and nothing else. <|eot_id|><|start_header_id|>user<|end_header_id|>
    '''

    suffix = ''
    suffix31 = r'''
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    '''

    txtfiles_dir = datadir + "/" + subdir + "/body/"
    # case_filenames = [f for f in os.listdir(txtfiles_dir) if f.endswith(".txt.gz")]
    # case_filenames = [f for f in os.listdir(txtfiles_dir) if f.endswith(".txt")]
    with open(txtfiles_dir + 'grepcase/found.txt', 'r') as file:
        case_filenames = [os.path.basename(f) for f in file.read().split("\n") if f != ""]
    random.shuffle(case_filenames)
    out_dir = txtfiles_dir + "../anns/"
    out_dir_iscase = out_dir + "iscase/"
    if out_dir == txtfiles_dir:
        exit
    os.makedirs(out_dir_iscase, exist_ok=True)

    # MAX_NUM = 10000

    # case_filenames = case_filenames[:MAX_NUM]

    # print(case_filenames) 

    batch_size = 1
    max_input_length=16384
    max_new_tokens = 256

    case_fn_batch = []
    case_txt_batch = []

    if True:
        out_dir_files = os.listdir(out_dir_iscase)  # check at beginning for files completed
        for idx, case in enumerate(case_filenames):
            # if idx < 18:
            #     continue
            if idx % 5 == 0:
                out_dir_files = os.listdir(out_dir_iscase)  # check at beginning for files completed
            if os.path.splitext(case)[0]+".csv" in out_dir_files:
                print('Skipping ', idx, "/", len(case_filenames), ": ", case, ". Already annotated")
                continue
            print(idx, "/", len(case_filenames), ": ", case)

            # Open file
            try:
                with gzip.open(txtfiles_dir + case, 'rt') as file:
                # with open(txtfiles_dir + case, 'r') as file:
                    case_txt = file.read()
                    num_lines = case_txt.count("\n")
                    case_txt = case_txt.replace('\n','')
            except:
                print('exception')
                with gzip.open(txtfiles_dir + case, 'rt', encoding='unicode_escape') as file:
                # with open(txtfiles_dir + case, 'r', encoding='unicode_escape') as file:
                    case_txt = file.read()
                    num_lines = case_txt.count("\n")
                    case_txt = case_txt.replace('\n','')

            if num_lines > 500:
                print(f'{case} is overlength ({num_lines}); skipping')
                continue
            case_txt_batch.append(case_txt)
            case_fn_batch.append(case)

            if len(case_txt_batch) < batch_size and idx != len(case_filenames):
                continue

            # Pass query to API
            # query = prefix + case_txt + suffix
            queries = [prefix31 + ct + suffix31 for ct in case_txt_batch]

            print('Passing batch to pipeline: ', ', '.join(case_fn_batch))
            
            if batch_size > 1:
                # try: 
                # sequences = pipeline(
                #     queries,
                #     do_sample=True,
                #     top_k=1,
                #     eos_token_id=tokenizer.eos_token_id,
                #     max_new_tokens = 8092,
                # )
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenized_batch = tokenizer(
                    queries,
                    padding=True,          # Pads all sequences to the same length
                    padding_side='left',
                    truncation=True,       # Truncates sequences longer than `max_length`
                    # max_new_tokens=max_new_tokens,         # Maximum sequence length
                    return_tensors="pt"    # Returns PyTorch tensors
                ).to(device)

                start_time = time.perf_counter()
                generated_ids = model.generate(
                    input_ids=tokenized_batch["input_ids"],
                    attention_mask=tokenized_batch["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                )
                end_time = time.perf_counter()
                total_time = end_time - start_time
                print(f'Generation took {total_time:.4f} seconds')

                # Decode generated outputs
                # sequences = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
                streams = tokenizer.batch_decode(generated_ids[:,tokenized_batch['input_ids'].shape[1]:], skip_special_tokens=True)

            if batch_size==1:
                inputs = tokenizer(
                    queries[0],
                    truncation=True,
                    max_length=max_input_length,
                    return_tensors="pt",
                )

                sequences = pipeline(
                    tokenizer.decode(inputs["input_ids"][0]),  # Decode back to text for pipeline,
                    do_sample=True,
                    top_k=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens = max_new_tokens,
                )
                for seq in sequences:
                    # print(f"Result: {seq['generated_text'][len(prefix)+len(case):]}")
                    stream = seq['generated_text'][len(queries[0]):]

                streams = [stream]

            # for seq in sequences:
            #     # print(f"Result: {seq['generated_text'][len(prefix)+len(case):]}")
            #     stream = seq['generated_text'][len(query):]

            # streams = [sequence['generated_text'][len(query):] for sequence, query in zip(sequences,queries)]
                # stream = seq['generated_text'][len(prefix31 + case_txt + suffix31):]

            # stream = client.chat.completions.create(
            #     #    model="gpt-4-turbo",
            #     model="o1-preview",
            #     messages=[
            # #        {"role": "system",
            # #               "content": system_prompt},
            #         {"role": "user",
            #             "content": base_prompt + "\n Apply the instructions above to the following query. Query: " + case_txt}
            #             ],
            # #    temperature=0,
            #     stream=False,
            # )
            # except:
            #     print(case, "did not work")
            #     continue

            # Write to out_dir
            for case_fn, stream in zip(case_fn_batch, streams):
                # with gzip.open(out_dir + os.path.splitext(case_fn)[0]+".bsv.gz", "wt") as text_file:
                with open(out_dir_iscase + os.path.splitext(case)[0]+".csv", "w") as text_file:
                    text_file.write(stream)

                # # Re-write as named csv
                # # out_names = [f for f in os.listdir(out_dir) if f.endswith(".bsv")]
                # # for outname in out_names:
                # try:
                #     df = pd.read_csv(out_dir + os.path.splitext(case_fn)[0]+".bsv.gz", sep="|", names = ("event","time"))
                
                #     df.to_csv(out_dir + os.path.splitext(case_fn)[0]+".csv.gz", index=False)
                # except:
                #     print(case_fn, "could not convert to csv")

            case_txt_batch = []
            case_fn_batch = []

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for model configuration.")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Name of the model to use."
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        # default="/data/<YOUR_DIR>/.cache/huggingface/hub/",
        help="Directory for caching model files."
    )
    
    parser.add_argument(
        "--datadir",
        type=str,
        # default="",
        help="Data directory name."
    )

    parser.add_argument(
        "--subdir",
        type=str,
        default="PMC001xxxxxx",
        help="Subdirectory name."
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(f"Model Name: {args.model_name}")
    print(f"Cache Directory: {args.cache_dir}")
    print(f"Subdirectory: {args.subdir}")
    print(f"Data directory: {args.datadir}")
    run(args)
