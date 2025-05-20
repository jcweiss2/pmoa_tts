from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import gzip
import random
import argparse
import time

def run(args):
    ### You just ran get_iscase1.sh and apply_llm_iscase_annotate_pmoa before this
    model_name = args.model_name
    cache_dir = args.cache_dir
    batch_size = args.batch_size
    subdir = args.subdir
    datadir = args.datadir

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                cache_dir=cache_dir,
                                                torch_dtype=torch.bfloat16,
                                                device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            cache_dir=cache_dir)

    device = 'cuda'
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
    You are an expert physician.  Output the list of diagnoses that pertain to the patient in the case report. For example, here is the case report.\
    An 18-year-old male was admitted to the hospital with a 3-day history of fever and rash. Four weeks ago, he was diagnosed with acne and received the treatment with minocycline, 100 mg daily, for 3 weeks. With increased WBC count, eosinophilia, and systemic involvement, this patient was diagnosed with DRESS syndrome. The fever and rash persisted through admission, and diffuse erythematous or maculopapular eruption with pruritus was present. One day later the patient was discharged.\
Then the output should look like\
    acne \
    DRESS syndrome \
    Rash, leukocytosis, and other findings are not diseases so they are omitted, while acne and DRESS syndrome were both diagnosed, so they are included.
    Output a list of diagnoses (one per line) from the following case. Place the primary diagnosis first. Include only the list and nothing else.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    '''

    suffix = ''
    suffix31 = r'''
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    '''

    txtfiles_dir = datadir +"/" + subdir + "/body/"
    # case_filenames = [f for f in os.listdir(txtfiles_dir) if f.endswith(".txt.gz")]
    # case_filenames = [f for f in os.listdir(txtfiles_dir) if f.endswith(".txt")]
    with open(txtfiles_dir + 'grepcase/found1.txt', 'r') as file:
        case_filenames = [os.path.basename(f).replace('.csv','.gz') for f in file.read().split("\n") if f != ""]
    random.shuffle(case_filenames)
    
    out_dir = txtfiles_dir + "../anns/"
    out_dir_dx = out_dir + "dx2/"
    if out_dir == txtfiles_dir:
        exit
    os.makedirs(out_dir_dx, exist_ok=True)

    # MAX_NUM = 10000

    # case_filenames = case_filenames[:MAX_NUM]

    # print(case_filenames) 

    max_new_tokens = 8092

    case_fn_batch = []
    case_txt_batch = []

    if True:
        out_dir_files = os.listdir(out_dir_dx)  # check at beginning for files completed
        for idx, case in enumerate(case_filenames):
            # if idx < 18:
            #     continue
            if idx % 5 == 0:
                out_dir_files = os.listdir(out_dir_dx)  # check at beginning for files completed
            if os.path.splitext(case)[0]+".bsv.gz" in out_dir_files:
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
                streams = tokenizer.batch_decode(generated_ids[:,tokenized_batch['input_ids'].shape[1]:], skip_special_tokens=True)

            if batch_size==1:
                sequences = pipeline(
                    queries[0],
                    do_sample=True,
                    top_k=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens = max_new_tokens,
                )
                for seq in sequences:
                    # print(f"Result: {seq['generated_text'][len(prefix)+len(case):]}")
                    stream = seq['generated_text'][len(queries[0]):]

                streams = [stream]

            # Write to out_dir
            for case_fn, stream in zip(case_fn_batch, streams):
                with gzip.open(out_dir_dx + os.path.splitext(case_fn)[0]+".bsv.gz", "wt") as text_file:
                # with open(out_dir_ts + os.path.splitext(case)[0]+".csv", "w") as text_file:
                    text_file.write(stream)

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
        "--subdir",
        type=str,
        default="PMC001xxxxxx",
        help="Subdirectory name."
    )

    parser.add_argument(
        "--datadir",
        type=str,
        # default="<DATA_DIR HERE>",
        help="Data directory name."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(f"Model Name: {args.model_name}")
    print(f"Cache Directory: {args.cache_dir}")
    print(f"Subdirectory: {args.subdir}")
    print(f"Data directory: {args.datadir}")
    print(f"Batch size: {args.batch_size}")
    run(args)