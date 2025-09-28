import torch
from test2 import LSTM, MIDIDataset, all_tokens, write_midi_file, token_to_id, id_to_token
from data_manager import find_all_midi_files


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)


def train():
    print(f"Using device: {torch.get_default_device()}")
    midi_files = find_all_midi_files('datasets/MAESTRO/data')

    dataset = MIDIDataset(midi_files)

    model = LSTM("lstm2.0", embedding_dim=16, hidden_size=256, dropout=0.2)
    model = model.to(torch.get_default_device())
    
    model.launch_training(dataset, epochs=200, batch_size=128, lr=0.001)
    print("Training complete.")

def generate():

    model = LSTM1("lstm1.2", embedding_dim=16, hidden_size=256)

    model.load_state_dict(torch.load('checkpoints/lstm1.2_epoch44.pth', map_location=torch.get_default_device())['model_state_dict'])

    generated = model.generate_stochastic_sequence(seq_len=512+1, temperature=.1)

    generated_tokens = [all_tokens[i] for i in generated]
    print("Generated tokens:", generated_tokens)
    print("Generated sequence:", generated)
    write_midi_file(generated[1:], "generated/test3.mid")
    print()


def debug():
    import matplotlib.pyplot as plt
    model = LSTM1("lstm1.2", embedding_dim=16, hidden_size=256)
    model = model.to(torch.get_default_device())
    model.load_state_dict(torch.load('checkpoints/lstm1.2_epoch44.pth', map_location=torch.get_default_device())['model_state_dict'])

    generated = [token_to_id["BOS"]]
    input_seq = torch.tensor([generated], dtype=torch.long, device=torch.get_default_device())


    while(True):
        output_probas = model.predict(input_seq, temperature=0.1)
        plt.bar(all_tokens, output_probas[0,-1].cpu().detach().numpy())
        plt.show()

        next_token = input("Enter next token (or 'exit' to stop): ")
        if next_token == 'EOS':
            break
        if next_token not in token_to_id:
            print("Invalid token. Try again.")
            continue
        generated.append(token_to_id[next_token])
        input_seq = torch.tensor([generated], dtype=torch.long, device=torch.get_default_device())

    print()


train()