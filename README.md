# Medi-CoQA
Conversational Question Answering on Clinical Text

## Docker
You can find this `Dockerfile` in the repository files. To reproduce the environment you can build a docker image locally and run the code inside docker.

Note: This is a GPU based image.
```
## build image
docker build -t transformers-coqa .

## run the iamge
docker run -it coqa

## run the code
cd transformer-coqa && \
. run.sh
```

## Non-Docker Alernative

# install packages 
run `pip install -r requirements.txt`

# install language model for spacy
python3 -m spacy download en
```

## Data

`coqa-train-v1.0.json` for training,
`coqa-dev-v1.0.json` for evaluating.
Download dataset from [CoQA](https://stanfordnlp.github.io/coqa/)

## Run-train

1. place coqa-train-v1.0.json and coqa-dev-v1.0.json in same folder, e.g. `data/`
2. run code using the command `./run.sh` in terminal
3. or run `run_coqa.py`
    ```bash
    python3 run_coqa.py --model_type albert \
                   --model_name_or_path albert-base-v2 \
                   --do_train \
                   --do_eval \
                   --data_dir data/ \
                   --train_file coqa-train-v1.0.json \
                   --predict_file coqa-dev-v1.0.json \
                   --learning_rate 3e-5 \
                   --num_train_epochs 2 \
                   --output_dir albert-output/ \
                   --do_lower_case \
                   --per_gpu_train_batch_size 8  \
                   --max_grad_norm -1 \
                   --weight_decay 0.01
                   --threads 8
                   --fp16

    ```

4. Shell script to reproduce models; i.e. ClinicalBERT :

   ```bash
   # Clinicalbert
   . ./run_clinicalbert.sh
   # albert
   . ./run.sh
   ```

5. The estimate training and evaluation time for `albert-base`, 'run_clinicalbert' model on the CoQA task is around **3** hours on AWS server g4dn.2xlarge.
## Run-evaluate

After you get the prediction files, you can evaluate on your test set seperately.
The evaluation script is provided by CoQA.
To evaluate, run

```bash
python3 evaluate.py --data-file <path_to_dev-v1.0.json> --pred-file <path_to_predictions>
```



## Results

Some commom parameters:
`adam_epsilon=1e-08, data_dir='data/', do_lower_case=True, doc_stride=128,  fp16=True, history_len=2, learning_rate=3e-05, max_answer_length=30, max_grad_norm=-1.0, max_query_length=64, max_seq_length=512,  per_gpu_eval_batch_size=8, seed=42, train_file='coqa-train-v1.0.json', warmup_steps=2000, weight_decay=0.01,num_train_epochs=2, threads=8`

Best results:
| Model                             | Em       | F1       | Parameters                                                   |
| --------------------------------- | -------- | -------- | ------------------------------------------------------------ |
| albert-base-v2                    | 71.5     | 81.0     | per_gpu_train_batch_size=8                                   |
| ClinicalBERT                      | 63.8     | 73.7     | per_gpu_train_batch_size=8                                   |


## Parameters

Here we will explain some important parameters, for all trainning parameters, you can find in `run_coqa.py`

| Param name                  | Default value        | Details                                                      |
| --------------------------- | -------------------- | ------------------------------------------------------------ |
| model_type                  | None                 | Type of models e.g. ClinicalBERT, ALBERT                  |
| model_name_or_path          | None                 | Path to pre-trained model or model name listed above.        |
| output_dir                  | None                 | The output directory where the model checkpoints and predictions will be written. |
| data_dir                    | None                 | The directory where training and evaluate data (json files) are placed, if  is None, the root directory will be taken. |
| train_file                  | coqa-train-v1.0.json | The training file.                                     |
| predict_file                | coqa-dev-v1.0.json   | The evaluation file.                                   |
| max_seq_length              | 512                  | The maximum total input sequence length after WordPiece tokenization. |
| doc_stride                  | 128                  | When splitting up a long document into chunks, stride length to take between chunks. |
| max_query_length            | 64                   | The maximum number of tokens for the question. Questions longer than this will be truncated to this length. |
| do_train                    | False                | Whether to run training.                                     |
| do_eval                     | False                | Whether to run evaluation on the dev set CoQA.                          |
| evaluate_during_training    | False                | Run evaluation during training at 10times each logging step  |
| do_lower_case               | False                | Set this flag if you are using an uncased model.             |
| per_gpu_train_batch_size    | 8                    | Batch size per GPU/CPU for training.                         |
| learning_rate               | 3e-5                 | The initial learning rate for Adam.                          |
| gradient_accumulation_steps | 1                    | Number of updates steps to accumulate before performing a backward/update pass. |
| weight_decay                | 0.01                 | Weight decay (optional to change).                               |
| num_train_epochs            | 2                    | Total number of training epochs to perform.                  |
| warmup_steps                | 2000                 | Linear warmup over warmup_steps.This should not be too small(such as 200), may lead to low score in model. |
| history_len                 | 2                    | keep len of history quesiton-answers                         |
| logging_steps               | 50                   | Log every X updates steps.                                   |
| threads                     | 8                    | no. of CPU's put for parallel processing          |
| fp16                        | True                 | half precision floating point format 16 bits; increases speed of model training.       |

## Model explanation

The following is the overview of the whole repo structure, we keep the structure similiar with the `transformers` fine-tune on `SQuAD`, we use the `transformers` library to load pre-trained model and model implementation.

```bash
├── data
│   ├── coqa-dev-v1.0.json  # CoQA Validation dataset
│   ├── coqa-train-v1.0.json # CoQA training dataset
│   ├── metrics
│   │   └── coqa_metrics.py # Compute and save the predictions, do evaluation and get the final score
│   └── processors
│       ├── coqa.py # Data processing: create examples from the raw dataset, convert examples into features
│       └── utils.py # data Processor for sequence classification data sets.
├── evaluate.py # script used to run the evaluation only, please refer to the above Run-evaluate section
├── model
│   ├── Layers.py # Multiple LinearLayer class used in the downstream QA tasks
│   ├── modeling_albert.py # core ALBERT model class, add architecture for the downstream QA tasks on the top of pre-trained ALBERT model from transformer library.
│   ├── modeling_auto.py # generic class that help instantiate one of the question answering model classes, As the bert like model has similiar input and output. Use this can make clean code and fast develop and test. Refer to the same class in transformers library
│   ├── modeling_bert.py # core BERT model class, includes all the architecture for the downstream QA tasks
├── README.md # This instruction you are reading now
├── requirements.txt # The requirements for reproducing our results
├── run_coqa.py # Main function script
├── run.sh # run training with default setting/ALBERT
├── run_clinicalbert.sh # run training with default setting/ClinicalBERT
└── utils
    └── tools.py # function used to calculate model parameter numbers
```

The following are detailed descrpition on some core scripts:

- [run_coqa.py](run_coqa.py): This script is the main function script for training and evaluation.
   1. Defines All system parameters and few training parameters
   2. Setup CUDA, GPU, distributed training and logging, all seeds
   3. Instantiate and initialize the corresponding model config, tokenizer and pre-trained model
   4. Calculate the number of trainable parameters
   5. Define and execute the training and evaluation function
- [coqa.py](data/processors/coqa.py): This script contains the functions and classes used to conduct data preprocess,
   1. Define the data structure of `CoqaExamples`, `CoqaFeatures` and `CoqaResult`
   2. Define the class of `CoqaProcessor`, which is used to process the raw data to get examples. It implements the methods `get_raw_context_offsets` to add word offset, `find_span_with_gt` to find the best answer span, `_create_examples` to convert single conversation (context and QAs pairs) into `CoqaExample`, `get_examples` to parallel execute the create_examples
   3. Define the methods `coqa_convert_example_to_features` to convert `CoqaExamples` into `CoqaFeatures`, `coqa_convert_examples_to_features` to parallel execute `coqa_convert_example_to_features`
- [modeling_albert.py](model/modeling_albert.py): This script contains the core ALBERT class and related downstream CoQA architecture,
   1. Load the pre-trained ALBERT model from `transformer` library
   2. Build downstream CoQA tasks architecture on the top of ALBERT last hidden state and pooler output to get the training loss for training and start, end, yes, no, unknown logits for prediction.

## References

1. [coqa-baselines](https://github.com/stanfordnlp/coqa-baselines)
2. [transformers](https://github.com/huggingface/transformers)
3. [bert4coqa](https://github.com/adamluo1995/Bert4CoQA)
4. [transformers-coqa](https://github.com/NTU-SQUAD/transformers-coqa)
