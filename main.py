import torch
from test1 import LSTM1, MIDIDataset1, all_tokens, write_midi_file
from data_manager import find_all_midi_files
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

torch.manual_seed(555)
torch.cuda.manual_seed_all(555)


def main():
    print(f"Using device: {torch.get_default_device()}")
    midi_files = find_all_midi_files('datasets/MAESTRO/data')

    dataset = MIDIDataset1(midi_files)
    
    model = LSTM1(embedding_dim=16, hidden_size=2048)
    model = model.to(torch.get_default_device())
    
        
    model.launch_training(dataset, epochs=20, batch_size=64, lr=0.001)
    print("Training complete.")

def generate(model):
    
    model.eval()

    generated = model.generate_valid_sequence(seq_len=128+1)
    print("Generated sequence:", generated)
    write_midi_file(generated[1:], "generated/test.mid")
    print()

main()