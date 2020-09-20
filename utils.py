import torch
import os


def test_batch(src_field, trg_field, translator, batch, device, max_examples=32):
    with torch.no_grad():
        source = batch.src.to(device)
        target = batch.trg.to(device)
        translator_batch = torch.argmax(translator(source, target.shape[0] - 1), dim=2)
    for i, (de_example, en_example, translator_example) in enumerate(
            zip(batch.src.split(1, dim=1), batch.trg.split(1, dim=1), translator_batch.split(1, dim=1))):
        if i >= max_examples:
            break
        print("source:")
        print(" ".join([src_field.vocab.itos[index] for index in de_example]).encode('utf-8'))

        print("target:")
        print(" ".join([trg_field.vocab.itos[index] for index in en_example]).encode('utf-8'))

        print("translation:")
        print(" ".join([trg_field.vocab.itos[index] for index in translator_example]).encode('utf-8'))
        print("\n\n")
        del de_example, en_example, translator_example
    del source, target, translator_batch


def load_model(model, checkpoint_folder):
    epoch = 0
    if os.path.exists(checkpoint_folder):
        while True:
            if os.path.exists(os.path.join(checkpoint_folder, f"epoch_{epoch}.pth")):
                epoch += 1
            else:
                break
        epoch -= 1
        model.load_state_dict(torch.load(os.path.join(checkpoint_folder, f"epoch_{epoch}.pth")))
    return epoch


def save_model(model, checkpoint_folder, epoch):
    model_file = os.path.join(checkpoint_folder, f"epoch_{epoch}.pth")
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    torch.save(model.state_dict(), model_file)