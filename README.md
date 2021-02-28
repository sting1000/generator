
# 2-layer Cascade Post-processor
## About The Project

This project implement a 2-layer cascade post-processor for Automatic Speech Recognition (ASR) system.
The purpose of post-processor is to format the spoken-style text, such as:
```
a baby giraffe is six feet tall and  weighs one hundred fifty pounds
```
to the corresponding written-style text:
```
a baby giraffe is 6ft tall and  weighs 150lb
```
This cascade post-processor contains two main modules: Classifier and Normalizer. 
Classifier assigns label to each token, which is similar to Named Entity Recognition (NER). 
The labeled sequences that require normalization are sent to Normalizer for conversion.
Normalizer employs Neural Network (RNN structure) and learn transform rules from dataset. 
Thus, the interest sequences are converted to written form, and put but to original sentence as output.

![Pipeline]()

Beside the intelligence, one big feature is customizablility. You can free to combine classifier model and normalizer
from various choices. For Classifier, pretrained models can be chosen from DistilBert, Bert, XLM, etc. 
For RNN Normalizer, Bi-LSTM, Transformer, GRU are all available to use. Moreover, some
cutting-edge tricks like copy mechanism can be easy to add through the configure file. 

## High Level Overview
![High Level]()

## Getting Started
### Step 0: Installation
1. Clone the repo
   ```sh
   git clone <repo_name.git>
   ```
2. Clone OpenNMT to repository root
   ```sh
   cd <repo_name>
   git clone https://github.com/OpenNMT/OpenNMT-py.git
   ```
3. Install requirements
   ```sh
   pip install requirements.txt
   pip install *.whl
   ```
### Step 1: Prepare dataset
To get started, we propose to generate speech command dataset for processor pipeline. 
This dataset can made by permutation of `./dataset/templates.json` and `./dataset/entities.json`.

* `templates.json` contains 4793 command templates, in 39 different intent and 4 languages. 
* `entities.json` provides 63k entities in 4 language to replace corresponding placeholder in templates.

We need to build a **JSON**  file to specify the path, language and range settings:

```json
{
  "templates_file": "data/templates.json",
  "entities_file": "data/entities.json",
  "languages": ["de", "en", "fr", "it"],
  "max_combo_amount": 3000,
  "max_channel_range": 3000,
  "mix_entity_types": ["VodName", "SeriesName", "BroadcastName"]
}
```
In the configuration, `languages` set which language(s) to use. 
`max_combo_amount` limits generated commands from a single template permutation process.
`mix_entity_types` indicate these types can be free to switch during permutation.

A simple configration `./config/prepare.json` is provided, and it will be necessary to train the model.

```sh
   python prepare.py --config ./config/prepare.json --prepared_dir ./output --valid_ratio 0.1 --test_ratio 0.1
```

This command will create a folder `./output`, including:
* `meta.csv`
* `train.csv`
* `validation.csv`
* `test.csv`

Train, validation, test are splited from meta dataset according to the options `--valid_ratio` and `--test_ratio`.
As default, the datasets are token level labeled with out padding. 
The script also provide option `----no_tagging` and `--padding` to customize the datasets. 
For specific info, please run `python prepare.py  -h`.

### Step 2: Build pipelines
Firstly, we need to define our pipeline model, the classifier and normalizer need to be configured separately. 

A CPU friendly dummy pipeline (distilbert-base + Bi-LSTM) is set as default and ready to go
(the training epochs are set to very small, so it is just for testing process). 

After finishing preparation, run the command below to build pipeline directory:

```sh
python build.py --pipeline_dir ./output/pipeline/distilbert-base_LSTM --prepared_dir ./output
```

Let's see what augments we can use here:

#### 1. Pipeline augments
* `--pipeline_dir`: str, Directory to save pipeline data and results
* `--prepared_dir`: str, Prepared dataset directory (containing test.csv, train.csv, validation.csv)

#### 2. Classifier augments
Since we use the pretrained models, the structure details do not need to change. Just give the pretrained source is enough.
Pretrained classifier can be downloaded online from [Huggingface model hub](https://huggingface.co/models) or use a local checkpoint. 

The trainer are configured as save the best model to `classifier_dir`,
 and other checkpoints are available in folder `classifier_dir/exp/`  

* `--classifier_dir`: str, Directory to save classifier model and data
* `--pretrained`: str, Load online model name or local pretrained dir. 

**Note**: 
* Set `--pretrained` as **None** to disable classifier, and the pipeline use normalizer only

  
#### 3. Normalizer augments
Normalizer requires network structure from *YAML* file as [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) described.
You can add more options like [here](https://opennmt.net/OpenNMT-py/options/train.html) or see 
[other example configurations](https://github.com/OpenNMT/OpenNMT-py/tree/master/config) for more complex normalizers.

In `./config` folder, RNN structures like *GRU*, *LSTM* and *Transformer* are provided. 
They works as model templates. All `{*PATH}` tokens in yaml template will be replace by `--normalizer_dir` during training,
and the replaced yaml file will be copied into `normalizer_dir/` with a same name.

All Normalizer's checkpoints will be saved to folder `normalizer_dir/checkpoints`

* `--normalizer_dir`: str, directory to save normalizer model and data
* `--model_yaml`: str, yaml model file path. 
* `--encoder_level`: str, default as char, [char, token]
* `--decoder_level`: str, default as char, [char, token]
* `--language`:str, language of the dataset [de, fr, en, it] (used only for Rule-based normalizer)
* `--onmt_dir`:str, OpenNMT package location

**Note**:
* Set `--model_yaml` as **None** to use Swisscom Rule-Based normalizer (VPN needed during training)


### Step 3: Train
Before running, make sure three things are ready: prepared datasets, pipeline folder, Internet connection.
Now, you can simply run the command to train the dummy pipeline `distilbert-base_LSTM`:

```sh
python train.py --pipeline_dir ./output/pipeline/distilbert-base_LSTM --mode pipeline --num_train_epochs 1
```
After running, the trained classifier and normalizer are saved to their designated directory.
The pipeline configuration are saved to `pipeline_dir` as `pipeline_args.txt`.
* `--pipeline_dir`: str, Directory to save pipeline data and results
* `--mode`: str, default as pipeline, train option in [pipeline, normalizer, classifier]
* `--num_train_epochs`: int, default as 10, epoch numbers to run classifier, and the best model will be saved
* `--per_device_train_batch_size`: int, default as 16, train batch size
* `--per_device_eval_batch_size`: int, default as 16, eval batch size
* `--learning_rate`: str,  default=1e-5, classifier learning rate
* `--weight_decay`: str, default=1e-2, classifier weight decay

 ### Step 3: Evaluation
Now, we would like to run pipeline evaluation on test dataset. 
Running command below, the WER/ WRR/ SER will be printed when it is done.

All results including classifier prediction, normalizer prediction and final output are saved to `pipeline_dir` for verbose reason.
The files have same prefix as `--key`.

```sh
python evaluate.py --pipeline_dir ./output/pipeline/distilbert-base_LSTM  --normalizer_step -1 --use_gpu=0
``` 

Here are the augments:
* `pipeline_dir`: str, trained pipeline dir including pipeline_args.txt
* `normalizer_step`: int, the training step of normalizer, -1 denotes as the last one
* `use_gpu`: int, default as 1, 1 to use gpu, 0 to use cpu
* `key`: str, default as test, choose dataset to evaluate [test, train, validation]

If you want to do evaluation between two txt files without loading models:
```shell script
wer ref.txt hyp.txt 
```
 ### Step 4: Predict
Finally, we can use the trained pipeline to do prediction. 
The input is txt file. One sentence per line. the prediction will go to output_path as a txt file.

If you run following this tutorial, the result of Step 3 and 4 will be bad 
since we just train very few steps on very few data. 

Run the command below to have the output txt file:
```sh
python predict.py --pipeline_dir ./output/pipeline/distilbert-base_LSTM --input_path ./example/input.txt --output_path ./example/output.txt --normalizer_step -1 --use_gpu=0
``` 
Here are the augments for `predict.py`:
* `pipeline_dir`: str, trained pipeline dir including pipeline_args.txt
* `input_path`: str, txt file that will be predicted
* `output_path`: str, output txt file path
* `normalizer_step`: int, the training step of normalizer, -1 denotes as the last one
* `use_gpu`: int, default as 1, 1 to use gpu, 0 to use cpu

## Pipelines from trained
Due to the feature of cascade, it is easy to reuse the trained classifier/ normalizer.
It is recommended to do experiments in Jupyter Notebook, 
so the `Normalizer` and `Classifier` can be easily called from the object `Pipeline` without reloading.
The API is similar: you can train/ eval/ pred them individually.

There is the steps to create new pipeline in Terminal:
1. Use `build.py`  with new `--normalizer_dir` and `--classifier_dir`  to create a new pipeline folder. 
Pay attention that settings like encoder/ decoder should be changed accordingly
2. Check if one/both of the module(s) is not trained. 
If so, run `train.py` which provides `--mode` option to train one module/ whole pipeline. 
2. Enjoy new pipeline! It can be used to `evaluate.py` and `predict.py` without extra training.  

## Other datasets
### Text Normalization Challenge
In [Text Normalization Challenge](https://www.kaggle.com/c/text-normalization-challenge-english-language/overview),
Google provided with a large EN corpus of text. Each sentence has a sentence_id. Each token within a sentence has a token_id. 
The before column contains the raw text, the after column contains the normalized text. 
The training set contains an additional column, class, which is provided to show the token type. 
This column is intentionally omitted from the test set. 
In addition, there is an id column used in the submission format. 
This is formed by concatenating the sentence_id and token_id with an underscore (e.g. 123_5).

After downloading the dataset, you can use the script `prepare_TNC.py` to generate a  prepared datasets folder like above.

### Museli
This is the dataset cannot be used for training pipeline since it does not have ground truth token label. 
But it is possible to use as input text and evaluate prediction quality by One Term Error Rate (OTER).
Here are some helpful scripts in `./src` to use:
* `make_museli_dataset.py` clean the responds from Jupyterhub and output a JSON file like this:
```json
{
    "intent": "SmartHomeSwitchOffDevice",
    "language": "de",
    "src": "gerät licht ausschalten",
    "entities_type": "SmartHomeDeviceName"
}
```
*`add_rb_to_json.py` is to add rule_based prediction to JSON file:
```json
{
    "intent": "SmartHomeSwitchOffDevice",
    "language": "de",
    "src": "gerät licht ausschalten",
    "entities_type": "SmartHomeDeviceName",
    "rb": "gerät licht ausschalten"
  }
```
*`calculate_oter.py` takes output predictions from processor, and sends predictions to **NLU** API for intent and entity_type.
Then calculate and save OTER results.

## Folder structure
    .
    ├── config                      # prepare configuration and YAML models
    ├── data                        # data sources (templates, entities, flaws)
    ├── example                     # contain example input.txt
    ├── images                      # repo readme figure
    ├── packages                    # .whl packages to install
    ├── src                         # scripts folder
    │   ├── Classifier.py           # class implementation for Classifier 
    │   ├── Normalizer.py           # class implementation for Normalizer
    │   ├── Pipeline.py             # class implementation for Pipeline
    │   ├── Processor.py            # converting written -> spoken
    │   ├── EntityCreator.py        # create extra numeric entities
    │   ├── SentenceGenerator.py    # permuation of template and entities
    │   ├── make_museli_dataset.py  # get museli dataset form hub
    │   ├── add_rb_to_json.py       # add rule-based prediction to json file
    │   ├── calculate_oter.py       # script to calculate oter 
    │   └── utils.py                # functions to help other classes
    ├── prepare.py                  # generate commands datasets to a folder
    ├── prepare_TNC.py              # process Text Normalization Channlenge datasets
    ├── build.py                    # make pipeline directory (configuration)
    ├── train.py                    # train the model(s) through pipeline dir
    ├── eval.py                     # eval pipeline processor
    ├── predict.py                  # predict given txt file
    ├── requirements.txt 
    ├── .gitignore
    └── README.md

## Low Level Overview
![Low Level]()