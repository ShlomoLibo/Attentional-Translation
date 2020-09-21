import torch
from argparse import ArgumentParser
import os

from model import Translator
from preprocess import get_data, EN_FIELD, DE_FIELD
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from train import BATCH_SIZE, DECODER_HIDDEN, ENCODER_HIDDEN, EMBEDDING_DIM_DECODER, EMBEDDING_DIM_ENCODER
from utils import load_model, test_batch

NUM_TEST = 500


def compare_validation(opt, iter_, num_test):
    input_dim = len(DE_FIELD.vocab)
    output_dim = len(EN_FIELD.vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sos_token = DE_FIELD.vocab.stoi["<SOS>"]
    translator = Translator(ENCODER_HIDDEN, DECODER_HIDDEN, input_dim, output_dim, EMBEDDING_DIM_ENCODER,
                            EMBEDDING_DIM_DECODER, sos_token=sos_token, device=device).to(device)
    cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=EN_FIELD.vocab.stoi["<PAD>"])

    epoch = 0
    print("testing...")
    if os.path.exists(opt.checkpoint_folder):
        while True:
            if os.path.exists(os.path.join(opt.checkpoint_folder, f"epoch_{epoch}.pth")):
                translator.load_state_dict(torch.load(os.path.join(opt.checkpoint_folder, f"epoch_{epoch}.pth")))
                with torch.no_grad():
                    total_loss = 0
                    for i, batch in enumerate(iter_):
                        if i < num_test:
                            source = batch.src.to(device)
                            target = batch.trg.to(device)[1:, :]  # without <SOS> token
                            output = translator(source, target.shape[0])
                            loss = cross_entropy(output.reshape(-1, output_dim), target.view(-1))
                            total_loss += loss.item()
                        else:
                            break
                    print(f"epoch {epoch}: {total_loss / num_test}")
                    epoch += 1
            else:
                break


def attention_vis(opt, iter_):
    input_dim = len(DE_FIELD.vocab)
    output_dim = len(EN_FIELD.vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sos_token = DE_FIELD.vocab.stoi["<SOS>"]
    translator = Translator(ENCODER_HIDDEN, DECODER_HIDDEN, input_dim, output_dim, EMBEDDING_DIM_ENCODER,
                            EMBEDDING_DIM_DECODER, sos_token=sos_token, device=device).to(device)
    load_model(translator, opt.checkpoint_folder)
    for batch in iter_:
        source = batch.src.to(device)
        target = batch.trg.to(device)[1:, :]  # without <SOS> token
        output, attention = translator.test(source, target.shape[0])
        test_batch(DE_FIELD, EN_FIELD, translator, batch, device=device)
        for i in attention.shape[0]:  # save attention table for each sample
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            cax = ax.matshow(attention[i].numpy(), cmap='bone')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + [DE_FIELD.vocab.itos[j] for j in source[:, i]] + ['<EOS>'], rotation=90)
            ax.set_yticklabels([''] + [EN_FIELD.vocab.itos[j] for j in source[:, i]])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            fig.savefig(os.path.join(opt.checkpoint_folder + "_attentions", f"attnetion_{i}.png"))
        input()
        del source, target, output, attention

def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_folder", type=str)
    parser.add_argument("--compare_validation", action="store_true", help="iterate through checkpoint folder and comapre on validation set")
    parser.add_argument("--attention_vis", action="store_true", help="generate attention tables of words in the dataset")
    opt = parser.parse_args()
    iter_train, iter_validate, iter_test = get_data(BATCH_SIZE)
    if opt.compare_validation:
        compare_validation(opt, iter_validate, NUM_TEST)
    if opt.attention_vis:
        attention_vis(opt, iter_test)


if __name__ == "__main__":
    main()
