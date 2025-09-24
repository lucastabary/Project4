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
    
    model = LSTM1("lstm1.2", embedding_dim=16, hidden_size=256)
    model = model.to(torch.get_default_device())
    
        
    model.launch_training(dataset, epochs=50, batch_size=128, lr=0.001)
    print("Training complete.")

def generate():

    model = LSTM1("lstm1.2", embedding_dim=16, hidden_size=256)

    model.load_state_dict(torch.load('checkpoints/lstm1_epoch20.pth', map_location=torch.get_default_device())['model_state_dict'])

    generated = model.generate_stochastic_sequence(seq_len=512+1, temperature=.5)
    print("Generated sequence:", generated)
    write_midi_file(generated[1:], "generated/test5.mid")
    print()

main()