import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pretty_midi
from data_manager import *
import numpy as np


special_tokens = ["PAD", "BOS", "EOS"]
# Pitch : 128 valeurs
pitch_tokens = [f"PITCH_{p}" for p in range(128)]

# Velocity : 16 buckets
velocity_buckets = [i*8 for i in range(16)]
velocity_tokens = [f"VELOCITY_{i}" for i in velocity_buckets]

# Durations : buckets hybrides (linéaire pour petits temps, log-spacés après)
duration_buckets = [0, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 2000]
duration_tokens = [f"DURATION_{d}" for d in duration_buckets]

# Delta_t : même logique que duration
delta_buckets = [0, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 2000]
delta_tokens = [f"DELTA_{d}" for d in delta_buckets]

# Assemblage
all_tokens = special_tokens + pitch_tokens + velocity_tokens + duration_tokens + delta_tokens

# Vocabulaire : token <-> id
token_to_id = {tok: i for i, tok in enumerate(all_tokens)}
id_to_token = {i: tok for i, tok in enumerate(all_tokens)}


def encode_pitch(p):
    return token_to_id[f"PITCH_{p}"]

def encode_velocity(v):
    idx = np.argmin([abs(v - b) for b in velocity_buckets])
    return token_to_id[f"VELOCITY_{velocity_buckets[idx]}"]

def encode_duration(d_ms):
    idx = np.argmin([abs(d_ms - b) for b in duration_buckets])
    return token_to_id[f"DURATION_{duration_buckets[idx]}"]

def encode_delta_t(dt_ms):
    idx = np.argmin([abs(dt_ms - b) for b in delta_buckets])
    return token_to_id[f"DELTA_{delta_buckets[idx]}"]

def write_midi_file(tokens, filename):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    current_time = 0.0
    i = 0
    while i < len(tokens):
        pitch_id = tokens[i]
        velocity_id = tokens[i+1]
        duration_id = tokens[i+2]
        delta_id = tokens[i+3]

        pitch = int(id_to_token[pitch_id].split('_')[1])
        velocity = int(id_to_token[velocity_id].split('_')[1])
        duration = int(id_to_token[duration_id].split('_')[1]) / 1000.0  # Convert ms to seconds
        delta_t = int(id_to_token[delta_id].split('_')[1]) / 1000.0  # Convert ms to seconds

        start_time = current_time + delta_t
        end_time = start_time + duration

        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
        piano.notes.append(note)

        current_time = start_time
        i += 4

    midi.instruments.append(piano)
    midi.write(filename)

def process_midi_file(midi_file):
    midi = pretty_midi.PrettyMIDI(midi_file)
    notes = []

    sorted_midi_notes = midi.instruments[0].notes
    sorted_midi_notes.sort(key=lambda x: x.start)  # Sort notes by start time

    last_start = sorted_midi_notes[0].start
    notes.append(token_to_id["BOS"])
    for note in sorted_midi_notes:
        start = note.start
        pitch = note.pitch
        velocity = note.velocity
        duration = note.end - note.start
        time_since_last = start - last_start
        notes.extend([encode_pitch(pitch), encode_velocity(velocity), encode_duration(duration), encode_delta_t(time_since_last)])
        last_start = start

    notes.append(token_to_id["EOS"])
    notes = torch.tensor(notes, dtype=torch.uint8).to(torch.get_default_device())
    return notes


class MIDIDataset1(Dataset):
    def __init__(self, midi_files, seq_len=2048):
        self.midi_files = midi_files
        self.seq_len = seq_len

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_data = process_midi_file(self.midi_files[idx])
        # Découpage aléatoire si la séquence est trop longue
        if len(midi_data) > self.seq_len:
            start = np.random.randint(0, len(midi_data) - self.seq_len)
            midi_data = midi_data[start:start+self.seq_len]
        return midi_data
        



class LSTM1(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super(LSTM1, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(len(all_tokens), embedding_dim=embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, len(all_tokens))

    def forward(self, x):
        x = self.embedding(x.long())
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out)
        return out
    
    def generate_sequence(self, seq_len=2048, device=None):
        if device is None:
            device = torch.get_default_device()
        generated = [token_to_id["BOS"]]
        input_seq = torch.tensor([generated], device=device)
        with torch.no_grad():
            for _ in range(seq_len - 1):
                if _ % 50 == 0: print(_)
                output = self(input_seq)
                next_token_logits = output[0, -1]
                next_token = torch.argmax(next_token_logits).item()
                generated.append(next_token)
                input_seq = torch.tensor([generated], device=device)
        return generated

    def generate_valid_sequence(self, seq_len=2048, device=None):
        if device is None:
            device = torch.get_default_device()
        generated = [token_to_id["BOS"]]
        input_seq = torch.tensor([generated], device=device)
        with torch.no_grad():
            wrong_tokens = 0
            for _ in range(seq_len - 1):
                output = self(input_seq)
                next_token_logits = output[0, -1]
                
                precedent_token = id_to_token[generated[-1]]
                if precedent_token.startswith("PITCH_"):
                    next_token = torch.argmax(next_token_logits[131:147]).item() + 131
                elif precedent_token.startswith("VELOCITY_"):
                    next_token = torch.argmax(next_token_logits[147:165]).item() + 147
                elif precedent_token.startswith("DURATION_"):
                    next_token = torch.argmax(next_token_logits[165:183]).item() + 165
                elif precedent_token.startswith("DELTA_"):
                    next_token = torch.argmax(next_token_logits[3:131]).item() + 3
                elif precedent_token == "BOS":
                    next_token = torch.argmax(next_token_logits[3:131]).item() + 3
                else:
                    print(f"Valid sequence finished early at length {_} due to invalid token {precedent_token}.")
                    return generated  # Stop if EOS or PAD
                generated.append(next_token)
                if next_token != torch.argmax(next_token_logits).item():
                    wrong_tokens += 1
                
                input_seq = torch.tensor([generated], device=device)
        
        print(f"{wrong_tokens} / {seq_len} wrong tokens ({(wrong_tokens/seq_len)*100:.2f}%)")
        return generated

    def quick_test(self, name):
        generated = self.generate_valid_sequence(seq_len=256+1)
        write_midi_file(generated[1:], f"generated/quicktest_{name}.mid")
        print(f"Quick test MIDI file saved as generated/quicktest_{name}.mid")

    def launch_training(self, dataset, epochs=10, batch_size=128, lr=0.001):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                generator=torch.Generator(device=torch.get_default_device()),
                                collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=token_to_id["PAD"]))
        criterion = nn.CrossEntropyLoss(ignore_index=token_to_id["PAD"])
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)

        print("Starting training...")
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                
                batch = batch.to(torch.get_default_device())

                targets = batch[:, 1:]  # Next token prediction
                inputs = batch[:, :-1]  # Align inputs with targets
                
                print(f"Inputs shape: {inputs.shape}, Input device : {inputs.device}")

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs.view(-1, len(all_tokens)), targets.contiguous().view(-1))
                loss.backward()
                optimizer.step()

                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            self.eval()
            self.quick_test(f"epoch{epoch+1}")
            self.train()

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoints/lstm1_epoch{epoch+1}.pth')