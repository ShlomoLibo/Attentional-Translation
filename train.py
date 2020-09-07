from argparse import ArgumentParser
from time import sleep

from utils import test_batch, load_model, save_model
from model import Translator
from preprocess import get_data, EN_FIELD, DE_FIELD
import torch
import os

BATCH_SIZE = 32
ENCODER_HIDDEN = 128
DECODER_HIDDEN = 128
EMBEDDING_DIM_ENCODER = 32
EMBEDDING_DIM_DECODER = 32


def main():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs to run the model")
    parser.add_argument("--checkpoint_folder", type=str, default="./checkpoints",
                        help="folder to save the model checkpoints to")
    parser.add_argument("--print_every", type=int, default=1000, help="how many batches between each report")
    opt = parser.parse_args()

    iter_train, iter_validate, iter_test = get_data(BATCH_SIZE)
    input_dim = len(DE_FIELD.vocab)
    output_dim = len(EN_FIELD.vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sos_token = DE_FIELD.vocab.stoi["<SOS>"]
    translator = Translator(ENCODER_HIDDEN, DECODER_HIDDEN, input_dim, output_dim, EMBEDDING_DIM_ENCODER,
                            EMBEDDING_DIM_DECODER, sos_token=sos_token, device=device).to(device)
    cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=EN_FIELD.vocab.stoi["<PAD>"])
    optimizer = torch.optim.Adam(translator.parameters(), lr=0.001)

    epoch = load_model(translator, opt.checkpoint_folder)
    total_iter = 0
    total_loss = 0
    print("running...")
    for epoch_ in range(epoch, epoch + opt.epochs):
        for i, batch in enumerate(iter_train):
            try:
                optimizer.zero_grad()
                source = batch.src.to(device)
                target = batch.trg.to(device)[1:, :]  # without <SOS> token
                output = translator(source, target.shape[0])
                loss = cross_entropy(output.reshape(-1, output_dim), target.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_iter += 1
                del source, target, output, loss
                if total_iter % opt.print_every == 0:
                    print(f"iter #{total_iter} loss: {total_loss / opt.print_every}")
                    total_loss = 0
                    test_batch_ = None
                    for j, batch_validate in enumerate(iter_validate):
                        test_batch_ = batch_validate
                        if j == total_iter / opt.print_every:
                            break
                    test_batch(DE_FIELD, EN_FIELD, translator, test_batch_, device=device)
                    del test_batch_
                    save_model(translator, opt.checkpoint_folder, epoch_)

            except Exception as e:
                print(e)
                sleep(120)


if __name__ == "__main__":
    main()


