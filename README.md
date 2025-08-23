## CoCoA (CoCoA: Confidence- and Context-Aware Adaptive Decoding for Resolving Knowledge Conflicts in Large Language Models)
Code for the paper [CoCoA: Confidence- and Context-Aware Adaptive Decoding for Resolving Knowledge Conflicts in Large Language Models].

by [Anant Khandelwal], [Manish Gupta], [Puneet Agrawal]

![image](https://github.com/user-attachments/assets/0df89574-1dd7-40f7-8187-7652e0ea05ed)

## Requirements
You can install all required packages by running the following command:
```bash
pip install -r requirements.txt
```

## Data
We used the same data provided by the authors of [AdaCAD](https://github.com/HanNight/AdaCAD). For sample download the dataset from AdaCAD repository to the data folder. 

## Run CoCoA
### For Question Answering
```bash
HF_TOKEN=your_huggingface_token # User Access Token to authenticate to the Hub.
HF_HUB_CACHE=your_cache_path # where repositories from the Hub will be cached locally (models, datasets and spaces).
bash run_qa.sh /path/to/your/input/file
```
As an exampe, run the following command:
```bash
bash run_qa.sh data/nq_swap_2_-1.json
```
We explain the arguments in `run_qa.sh` as follows:
- `GLOBALLEN`: the maximum sequence length of the model.
- `MAXCTXLEN`: the maximum input context length.
- `GENLEN`: the maximun generation length, should be `GENLEN = GLOBALLEN - MAXCTXLEN`.
- `SEED`: random seed.
- `DEVICE`: the GPU device ids, for example, `0,1`.
- `TOPP`: top-p sampling, set to 0.0 for greedy decoding.
- `GPUS`: number of gpus.
- `FLAG`: whether to use int4 quantization to load the model.

**Note:** Remember to use your own huggingface token and set your local cache path.

### For Summarization
```bash
HF_TOKEN=your_huggingface_token # User Access Token to authenticate to the Hub.
HF_HUB_CACHE=your_cache_path # where repositories from the Hub will be cached locally (models, datasets and spaces).
bash run_summ.sh /path/to/your/input/file
```
As an exampe, run the following command:
```bash
bash run_summ.sh tofu_1.5_-0.5.jsonl
```
The aguments are the same as those in `run_qa.sh`, except that the new argument `THRESHOLD` is added to set the threshold for the `alpha` as warmup operation for long-form generation.


## Acknowledgement
We sincerely thank the authors of [AdaCAD](https://github.com/HanNight/AdaCAD) for their public code release and providing all the created datasets and evaluation scripts by them.

## Citation
```bibtex

```
