# Attentional Translation
This project is a pytorch implementation of the [Neural Machine Tanslation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) paper, for German to English translation based on the Multi30k dataset.

## Requirements
Download the following ```.yml``` file for the required anaconda environment: [Link](https://drive.google.com/file/d/1DQL8cr8L7LDoAeVV3_22i3xvWn3S4chs/view?usp=sharing)
<br>
To create the environment run:
```
conda env create -f attn_translation_env.yml
```
To activate the environment run:
```
conda activate attn_translation_env
```

Additionally, while the environment is activated, download the ```spacy``` tokenizer by running:
```
pip install spacy
```
and then:
```
python -m spacy download en
python -m spacy download de
```

## Training
To train the model run the following command:
```
python train.py --checkpoint_folder <checkpoint_folder>
```
Where each epoch will be saved into ``` <checkpoint_folder>```.

## Testing
To test the model simply run:
```
python test.py --checkpoint_folder <checkpoint_folder> --attention_vis
```
Where ```<checkpoint_folder>``` is the folder used in the training phase.
This will iterate through batches in the testing dataset, where the attentional diagrams for each sentence in the batch will appear in  ```<checkpoint_folder>_attentions```, to go to the next batch press ```enter```.
<br>
To test the model across the different epochs on the validation dataset run:
```
python test.py --checkpoint_folder <checkpoint_folder> --compare_validation
```

## Pre-Trained Model
You can find a pre-trained model [here](https://drive.google.com/file/d/1jQbOz1J0WomIHeFAEoMGUY69edSAzlhi/view?usp=sharing). <br>
Simply save it under ```<checkpoint_folder>/epoch_0.pth``` (insure there are no other files in this directory).
