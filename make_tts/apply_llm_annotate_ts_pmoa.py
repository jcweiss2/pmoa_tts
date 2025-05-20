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
    You are a physician.  Extract the clinical events and the related time stamp from the case report. The admission event has timestamp 0. If the event is not available, we treat the event, e.g. current main clinical diagnosis or treatment with timestamp 0. The events happened before event with 0 timestamp have negative time, the ones after the event with 0 timestamp have positive time. The timestamp are in hours. The unit will be omitted when output the result. If there is no temporal information of the event, please use your knowledge and events with temporal expression before and after the events to provide an approximation. We want to predict the future events given the events happened in history. For example, here is the case report.\
    An 18-year-old male was admitted to the hospital with a 3-day history of fever and rash. Four weeks ago, he was diagnosed with acne and received the treatment with minocycline, 100 mg daily, for 3 weeks. With increased WBC count, eosinophilia, and systemic involvement, this patient was diagnosed with DRESS syndrome. The fever and rash persisted through admission, and diffuse erythematous or maculopapular eruption with pruritus was present. One day later the patient was discharged.\
    Let's find the locations of event in the case report, it shows that four weeks ago of fever and rash, four weeks ago, he was diagnosed with acne and receive treatment. So the event of fever and rash happen four weeks ago, 672 hours, it is before admitted to the hospital, so the time stamp is -672. diffuse erythematous or maculopapular eruption with pruritus was documented on the admission exam, so the time stamp is 0 hours, since it happens right at admission. DRESS syndrome has no specific time, but it should happen soon after admission to the hospital, so we use our clinical judgment to give the diagnosis of DRESS syndrome the timestamp 0. then the output should look like\
    18 years old| 0\
    male | 0\
    admitted to the hospital | 0\
    fever | -72\
    rash | -72\
    acne |  -672\
    minocycline |  -672\
    increased WBC count | 0\
    eosinophilia| 0\
    systemic involvement| 0\
    diffuse erythematous or maculopapular eruption| 0\
    pruritis | 0\
    DRESS syndrome | 0\
    fever persisted | 0\
    rash persisted | 0\
    discharged | 24\
    Separate conjunctive phrases into its component events and assign them the same timestamp (for example, the separation of 'fever and rash' into 2 events: 'fever' and 'rash')  If the event has duration, assign the event time as the start of the time interval. Attempt to use the text span without modifications except 'history of' where applicable. Include all patient events, even if they appear in the discussion; do not omit any events; include termination/discontinuation events; include the pertinent negative findings, like 'no shortness of breath' and 'denies chest pain'.  Show the events and timestamps in rows, each row has two columns: one column for the event, the other column for the timestamp.  The time is a numeric value in hour unit. The two columns are separated by a pipe '|' as a bar-separated file. Skip the title of the table. Reply with the table only. Create a table from the following case:
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    '''

    suffix = ''
    suffix31 = r'''
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    '''

    txtfiles_dir = datadir + "7/" + subdir + "/body/"
    # case_filenames = [f for f in os.listdir(txtfiles_dir) if f.endswith(".txt.gz")]
    # case_filenames = [f for f in os.listdir(txtfiles_dir) if f.endswith(".txt")]
    with open(txtfiles_dir + 'grepcase/found1.txt', 'r') as file:
        case_filenames = [os.path.basename(f).replace('.csv','.gz') for f in file.read().split("\n") if f != ""]
    random.shuffle(case_filenames)
    out_dir = txtfiles_dir + "../anns/"
    out_dir_ts = out_dir + "ts/"  # unused (forgot to point here)
    if out_dir == txtfiles_dir:
        exit
    os.makedirs(out_dir_ts, exist_ok=True)

    # MAX_NUM = 10000

    # case_filenames = case_filenames[:MAX_NUM]

    # print(case_filenames) 

    max_new_tokens = 8092

    case_fn_batch = []
    case_txt_batch = []

    if True:
        out_dir_files = os.listdir(out_dir)  # check at beginning for files completed
        for idx, case in enumerate(case_filenames):
            # if idx < 18:
            #     continue
            if idx % 5 == 0:
                out_dir_files = os.listdir(out_dir)  # check at beginning for files completed
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
                with gzip.open(out_dir + os.path.splitext(case_fn)[0]+".bsv.gz", "wt") as text_file:
                # with open(out_dir_ts + os.path.splitext(case)[0]+".csv", "w") as text_file:
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
        # default="/data/<YOUR-DIR>/.cache/huggingface/hub/",
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
        # default="",
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