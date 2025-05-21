from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import gzip
import random
import os
from tqdm import tqdm

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.bfloat16,  cache_dir="/data/CHARM-MIMIC/.cache/huggingface/hub/", device_map='auto', offload_folder="offload",  offload_state_dict=True) 

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/data/CHARM-MIMIC/.cache/huggingface/hub/")

pipeline = transformers.pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, eos_token_id=tokenizer.eos_token_id, tokenizer=tokenizer, device_map="auto")


prefix31 = r'''
<s> [INST]
You are a physician.  You are given a case report and an associated time series, and you need to determine if the patient dies and find the mention from the time series.  If the patient dies, return the row contents corresponding to the death event (the event and time from that row).  If the patiet does not die, set the event to 'censored', and the time to the maximum time in the time series.  For example, suppose you receive the case report.

An 18-year-old male was admitted to the hospital with a 3-day history of fever and rash. Four weeks ago, he was diagnosed with acne and received the treatment with minocycline, 100 mg daily, for 3 weeks. With increased WBC count, eosinophilia, and systemic involvement, this patient was diagnosed with DRESS syndrome.  The patient suddenly deterioriated from renal failure and expired on the 7th day.

And it's time series is:

18-year-old, 0
male, 0
fever, -72
rash, -72
acne, -672
minocycline, -672
increased WBC count, 0
eosinophilia, 0
systemic involvement, 0
DRESS syndrome, 0
renal failure, 168
expired, 168

Then, the correct repsonse is:

expired, 168

Because the row in the time series is 'expired, 168'.  If the time series row had been 'dead, 168', then replying with 'expired, 168' would be incorrect.  If the event did not happen, the correct response would be 'censored, 168'.
Output the single row only and nothing else. Provide your response to the following case report:
'''

interlude = r'''

Here is the associated time series:

'''

suffix = ''
suffix31 = r'''
[/INST]

'''

txtfiles_dir = '/data/CHARM-MIMIC/data/pmoa241217/Sayantan/sampled_25k_body/clean/'

annotations_dir = '/data/CHARM-MIMIC/data/pmoa241217/Sayantan/sampled_25k_timeord/'

case_filenames = [f for f in os.listdir(txtfiles_dir) if f.endswith(".txt")]
ann_filenames = [f for f in os.listdir(annotations_dir) if f.endswith(".csv")]

out_dir = '/data/CHARM-MIMIC/data/pmoa241217/Sayantan/temp_llm_death_phe_output/'


if out_dir == txtfiles_dir:
    exit
os.makedirs(out_dir, exist_ok=True)

MAX_NUM = 25000

case_filenames = case_filenames[:MAX_NUM]

random.shuffle(case_filenames)
#print(case_filenames) 

out_dir_files = os.listdir(out_dir)  # check at beginning for files completed


for idx, case in tqdm(enumerate(case_filenames)):
    case_ann = os.path.splitext(case)[0]+".csv"

    if not os.path.exists(annotations_dir + case_ann):
        print(f"No matching annotation file for {case}")
        continue

    # if idx % 5 == 0 and len(case_filenames) - len(out_dir_files) > 50:
    out_dir_files = os.listdir(out_dir) 
    # if idx < 18:
    #     continue
    if os.path.splitext(case)[0]+".bsv" in out_dir_files:
        print('Skipping ', idx, "/", len(case_filenames), ": ", case, ". Already annotated")
        continue
    print(idx, "/", len(case_filenames), ": ", case)



    # Open file
    try:
        # with gzip.open(txtfiles_dir + case, 'rt') as file:
        with open(txtfiles_dir + case, 'r') as file:
            case_txt = file.read().replace('\n','')

    except UnicodeDecodeError:
        # print('exception')
        # # with gzip.open(txtfiles_dir + case, 'rt', encoding='unicode_escape') as file:
        # with open(txtfiles_dir + case, 'r', encoding='unicode_escape') as file:
        #     case_txt = file.read().replace('\n','')

        print(f"Skipping {case}: decoding error")
        continue

    try:
        # with gzip.open(txtfiles_dir + case, 'rt') as file:
        with open(annotations_dir + case_ann, 'r') as file:
            ann_txt = file.read() #.replace('\n','')
    except:
        print('exception')
        # with gzip.open(txtfiles_dir + case, 'rt', encoding='unicode_escape') as file:
        with open(annotations_dir + case, 'r', encoding='unicode_escape') as file:
            ann_txt = file.read().replace('\n','')

    #Pass query to API
    #query = prefix + case_txt + suffix
    query = prefix31 + case_txt + interlude + ann_txt + suffix31

    # try: 
    sequences = pipeline(
        query,
        do_sample=True,
        top_k=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens = 8092,)
    
    for seq in sequences:
        # print(f"Result: {seq['generated_text'][len(prefix)+len(case):]}")
        stream = seq['generated_text'][len(query):]

    full_output_path = os.path.join(out_dir, os.path.splitext(case)[0]+".bsv")

    with open(full_output_path, "w") as text_file:
        text_file.write(stream)