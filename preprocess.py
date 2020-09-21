from typing import Dict, Union, List, Tuple, Optional, Iterator


from torchtext import data, datasets
EN_FIELD = data.Field(init_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>", lower=True, tokenize="spacy", tokenizer_language="en")
DE_FIELD = data.Field(init_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>", lower=True, tokenize="spacy", tokenizer_language="de")


def get_data(batch_size) -> Tuple[Iterator, Iterator, Iterator]:
    train_data, validation_data, test_data = datasets.Multi30k.splits(exts=(".de", ".en"), fields=(DE_FIELD, EN_FIELD))
    EN_FIELD.build_vocab(train_data, min_freq=2)
    DE_FIELD.build_vocab(train_data, min_freq=2)
    return data.BucketIterator.splits((train_data, validation_data, test_data), batch_size=batch_size)

