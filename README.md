# PMOA-TTS: PubMed Open Access Textual Time Series

📄 This repository accompanies the paper: [PMOA-TTS: Introducing the PubMed Open Access Textual Times Series Corpus](https://arxiv.org/abs/2505.20323).

## 📚 Overview

PMOA-TTS is the first large-scale, openly available dataset of 124,699 PubMed Open Access (PMOA) case reports, each converted into structured (event, time) timelines via a scalable LLM-based pipeline.

Our approach combines heuristic filtering with Llama 3.3 to identify single-patient case reports, followed by prompt-driven extraction using Llama 3.3 and DeepSeek R1, resulting in over 5.6 million timestamped clinical events.

This repository provides the codebase for timeline extraction, forecasting experiments, survival analysis, and evaluation.

## 📂 Repository Structure

- `make_tts/`: Scripts and notebooks for constructing the PMOA-TTS dataset from raw case reports.
- `forecasting_results/`: Code and results for forecasting experiments on the textual time series data.
- `survival_analyses/`: Tools and analyses for survival prediction tasks using the extracted timelines.
- `TTS_evaluation/`: Evaluation scripts assessing the quality of the extracted timelines.

Each subdirectory contains its own `README.md` with detailed instructions and information.

## 🛠️ Getting Started

1. **Clone the repository:**

   ```
   bash
   git clone https://github.com/jcweiss2/pmoa_tts.git
   cd pmoa_tts
   ```

2. **Set up the environment:**

   Ensure you have Python 3.10 or higher installed. It's recommended to use a virtual environment.

   ```
   conda create -n pmoa_tts python 3.10 
   conda activate pmoa_tts 
   conda install pandas scikit-learn numpy tensorboard -c conda-forge 
   conda install pytorch torchvision torchaudio -c pytorch 
   pip install transformers 
   pip install argparse 
   conda install jupyter -c conda-forge 
   pip install sentencepiece
   ```

3. **Access the PMOA-TTS dataset**
The dataset is available on [Hugging Face](https://huggingface.co/datasets/snoroozi/pmoa-tts). You can download it using the datasets library.

```
from datasets import load_dataset
dataset = load_dataset("snoroozi/pmoa-tts")
```

## 📄 Dataset Details

- **Number of Records:** Approximately 174,000 case reports  
- **Data Format:** Parquet files, compatible with Hugging Face Datasets and pandas  
- **Languages:** English  
- **License:** CC BY-NC-SA 4.0  
- **Tags:** clinical, time-series, biomedical, text  
- **Tasks Supported:** Text Classification, Time Series Forecasting  

Each data point includes:

- **Textual Time Series:** A sequence of timestamped clinical events  
- **Demographics:** Age, sex, and ethnicity (when available)  
- **Diagnoses:** Extracted diagnoses from the case report  
- **Death Information:** Observed time and death event indicator  

For more details, visit the [Hugging Face dataset page](https://huggingface.co/datasets/snoroozi/pmoa-tts).

## 📄 Citation

If you use PMOA-TTS in your research, please cite our preprints:

```bibtex
@article{noroozizadeh2025pmoa,
  title={PMOA-TTS: Introducing the PubMed Open Access Textual Times Series Corpus},
  author={Noroozizadeh, Shahriar and Kumar, Sayantan and Chen, George H. and Weiss, Jeremy C.},
  journal={arXiv preprint arXiv:2505.20323},
  year={2025}
}
   
