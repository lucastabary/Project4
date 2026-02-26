import torch
from test5 import LSTM, MIDIDataset, all_tokens, write_midi_file, token_to_id, id_to_token, process_midi_file
from data_manager import find_all_midi_files


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

midi_files = find_all_midi_files('datasets/MAESTRO/data')
dataset = MIDIDataset.load_processed('datasets/maestro-reduced.pt', seq_len=2**10)
# dataset.save_processed('datasets/maestro-reduced.pt')

model = LSTM("lstm5.1", embedding_dim=16, hidden_size=256, dropout=0.2)
model = model.to(torch.get_default_device())
# model.load_state_dict(torch.load('trainings/lstm4.1/lstm4.1_epoch6000.pth', map_location=torch.get_default_device())['model_state_dict'])

print(f"Using device: {torch.get_default_device()}")


def train():
    
    # Training on small sequences
    dataset.seq_len = 1024
    model.launch_training(dataset, epochs=6000, batch_size=128, lr=0.001)
    print("Training on small sequences complete.")

    # Training LoRA on long sequences
    dataset.seq_len = 2**15 # The average length of MIDI files in the dataset is around 22k tokens so 2**15=32768 should consider the whole files
    model.launch_training(dataset, epochs=2000, batch_size=16, lr=0.001)

def generate():
 
    model.eval()
    generated = model.generate_stochastic_sequence(seq_len=2**13 +1, temperature=0.8)

    print("Generated sequence:", generated)
    write_midi_file(generated[1:], "generated/test18.mid")
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

def analyse_embeddings():

    cos = lambda a,b : (a @ b) / ((a**2).sum()**0.5 * (b**2).sum()**0.5)
    
    import matplotlib.pyplot as plt

    embeddings = model.embedding.weight.cpu().detach().numpy()

    cosine_similarity = embeddings / (embeddings**2).sum(axis=1, keepdims=True)**0.5
    cosine_similarity = cosine_similarity @ cosine_similarity.T

    # Pitch similarities
    plt.figure(figsize=(10, 8))
    
    plt.title("Pitch Similarities")
    plt.xlabel("Token ID")
    plt.ylabel("Token ID")
    plt.imshow(cosine_similarity, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Cosine Similarity of Token Embeddings")
    plt.show()

    print()

train()