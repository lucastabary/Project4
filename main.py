import torch
from test1 import LSTM1, MIDIDataset1, all_tokens, train_model
from data_manager import find_all_midi_files
import time

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(555)
torch.cuda.manual_seed_all(555)

def main():

    print(f"Using device: {torch.get_default_device()}")


    midi_files = find_all_midi_files('datasets/MAESTRO/data')

    dataset = MIDIDataset1(midi_files)

    test = dataset[0]  # Get the first MIDI data
    
    model = LSTM1(embedding_dim=16, hidden_size=256)
    model = model.to(torch.get_default_device())
    
        
    train_model(model, dataset, epochs=20, batch_size=32, lr=0.001)
    print("Training complete.")

main()