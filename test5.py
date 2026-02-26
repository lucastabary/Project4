"""
L'idée est d'entrainer le LSTM sur de courtes séquences, puis d'entrainer un LoRA sur des séquences plus longues.
Le LSTM va apprendre les structures locales, et le LoRA les dépendances à long terme.
On implémente TBPTT pour l'entrainement du LSTM sur des séquences longues.
"""



import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pretty_midi
from data_manager import *
import numpy as np
import logging

logging.basicConfig(filename="logs/lstm_5_training.log",
                    format='%(asctime)s: %(levelname)s: %(message)s',
                    level=logging.INFO)

special_tokens = ["PAD", "BOS", "EOS"]
# Pitch : 128 valeurs
pitch_tokens = [f"PITCH_{p}" for p in range(21,109)]  # MIDI pitches from 21 (A0) to 108 (C8)

# Velocity : 16 buckets
velocity_buckets = [i*8 for i in range(16)]
velocity_tokens = [f"VELOCITY_{i}" for i in velocity_buckets]

# Durations : buckets hybrides (linéaire pour petits temps, log-spacés après)
duration_buckets = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 2000]
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
    while i < len(tokens) and id_to_token[tokens[i]] != "EOS":
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
        duration = (note.end - note.start) * 1000  # Convert to ms
        time_since_last = (start - last_start) * 1000  # Convert to ms
        notes.extend([encode_pitch(pitch), encode_velocity(velocity), encode_duration(duration), encode_delta_t(time_since_last)])
        last_start = start

    notes.append(token_to_id["EOS"])
    notes = torch.tensor(notes, dtype=torch.uint8).to('cpu')
    return notes


class MIDIDataset(Dataset):

    @classmethod
    def load_processed(cls, path, seq_len=2**14):
        all_midi_data = torch.load(path)
        dataset = cls([], seq_len=seq_len)
        dataset.all_midi_data = all_midi_data
        return dataset
    
    def __init__(self, midi_files, seq_len=2**14):
        self.midi_files = midi_files
        self.seq_len = seq_len
        
        print(f"Processing {len(midi_files)} MIDI files...")
        self.all_midi_data = [process_midi_file(f) for f in midi_files]
        print("Processing complete.")

    def __len__(self):
        return len(self.all_midi_data)

    def __getitem__(self, idx):
        midi_data = self.all_midi_data[idx]
        # Découpage aléatoire si la séquence est trop longue
        if len(midi_data) > self.seq_len:
            start = np.random.randint(0, len(midi_data) - self.seq_len - 1)
            midi_data = midi_data[start:start+self.seq_len+1]
        return midi_data
    
    def save_processed(self, path):
        torch.save(self.all_midi_data, path)
        print(f"Processed dataset saved to {path}")
    
    def count_tokens(self):
        count = {token: 0 for token in all_tokens}
        for midi_data in self.all_midi_data:
            for token in midi_data.tolist():
                count[id_to_token[token]] += 1
        return count


class LoRA_LSTM_Layer(nn.Module):
    def __init__(self, lstm_layer, r=4):
        super(LoRA_LSTM_Layer, self).__init__()
        self.lstm_layer = lstm_layer
        self.r = r

        # LoRA parameters for input-hidden weights
        self.W_ih_A = nn.Linear(lstm_layer.input_size, r, bias=False)
        self.W_ih_B = nn.Linear(r, lstm_layer.hidden_size * 4, bias=False)

        # LoRA parameters for hidden-hidden weights
        self.W_hh_A = nn.Linear(lstm_layer.hidden_size, r, bias=False)
        self.W_hh_B = nn.Linear(r, lstm_layer.hidden_size * 4, bias=False)
    
    def forward(self, x, hidden, use_lora=True):
        # Original LSTM computation
        h_0, c_0 = hidden
        output, (h_n, c_n) = self.lstm_layer(x, (h_0, c_0))
        if not use_lora:
            return output, (h_n, c_n)
        

        # LoRA adjustments
        W_ih_lora = self.W_ih_B(self.W_ih_A(x))
        W_hh_lora = self.W_hh_B(self.W_hh_A(h_0))

        # Combine original and LoRA weights
        adjusted_output = output + W_ih_lora + W_hh_lora

        return adjusted_output, (h_n, c_n)

class LSTM(nn.Module):
    def __init__(self, name, embedding_dim, hidden_size, dropout):
        super(LSTM, self).__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(len(all_tokens), embedding_dim=embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, len(all_tokens))

    def forward(self, x):
        x = self.embedding(x.long())
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dropout3(out)
        out = self.fc(out)
        return out
    
    def generate(self, seq_len=2048, device=None):
        return
    
    def predict(self, x, temperature=1.0):
        out = self.forward(x)
        return torch.softmax(out / temperature, dim=-1)

    def step(self, token, hidden_states=None, device=None):
        """Run a single-token forward pass keeping and returning LSTM hidden states.

        token: int token id
        hidden_states: tuple of ((h1,c1),(h2,c2),(h3,c3)) or None
        returns: logits (1D tensor of vocab size), new_hidden_states
        """
        if device is None:
            device = torch.get_default_device()

        # prepare input (batch=1, seq=1)
        token_t = torch.tensor([[token]], device=device, dtype=torch.long)
        emb = self.embedding(token_t)  # (1, 1, emb_dim)

        # init hidden states if necessary
        def _init_state():
            h = torch.zeros(1, 1, self.hidden_size, device=device)
            c = torch.zeros(1, 1, self.hidden_size, device=device)
            return (h, c)

        if hidden_states is None:
            h1, h2, h3 = _init_state(), _init_state(), _init_state()
        else:
            h1, h2, h3 = hidden_states

        out1, (h1_h, h1_c) = self.lstm1(emb, h1)
        out1 = self.dropout1(out1)
        out2, (h2_h, h2_c) = self.lstm2(out1, h2)
        out2 = self.dropout2(out2)
        out3, (h3_h, h3_c) = self.lstm3(out2, h3)
        out3 = self.dropout3(out3)

        logits = self.fc(out3)  # (1,1,vocab)
        logits = logits[0, -1]  # (vocab,)

        new_hidden_states = ((h1_h, h1_c), (h2_h, h2_c), (h3_h, h3_c))
        return logits, new_hidden_states

    def generate_sequence(self, seq_len=2048, device=None):
        if device is None:
            device = torch.get_default_device()
        generated = [token_to_id["BOS"]]
        input_seq = torch.tensor([generated], device=device)
        with torch.no_grad():
            for _ in range(seq_len - 1):
                output = self.predict(input_seq)
                next_token = torch.argmax(output[0, -1]).item()
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
            # compute token group boundaries programmatically
            n_special = len(special_tokens)
            n_pitch = len(pitch_tokens)
            n_velocity = len(velocity_tokens)
            n_duration = len(duration_tokens)
            n_delta = len(delta_tokens)

            pitch_start = n_special
            pitch_end = pitch_start + n_pitch
            velocity_start = pitch_end
            velocity_end = velocity_start + n_velocity
            duration_start = velocity_end
            duration_end = duration_start + n_duration
            delta_start = duration_end
            delta_end = delta_start + n_delta

            # incremental generation using hidden states to avoid reprocessing whole sequence
            hidden_states = None
            for _ in range(seq_len - 1):
                # compute logits for the next token using only the last token and hidden states
                logits, hidden_states = self.step(generated[-1], hidden_states=hidden_states, device=device)

                precedent_token = id_to_token[generated[-1]]
                if precedent_token.startswith("PITCH_"):
                    start, end = velocity_start, velocity_end
                elif precedent_token.startswith("VELOCITY_"):
                    start, end = duration_start, duration_end
                elif precedent_token.startswith("DURATION_"):
                    start, end = delta_start, delta_end
                elif precedent_token.startswith("DELTA_"):
                    start, end = pitch_start, pitch_end
                elif precedent_token == "BOS":
                    start, end = pitch_start, pitch_end
                else:
                    print(f"Valid sequence finished early at length {_} due to invalid token {precedent_token}.")
                    break

                slice_logits = logits[start:end]
                next_token_rel = torch.argmax(slice_logits).item()
                next_token = next_token_rel + start
                generated.append(next_token)
                if next_token != torch.argmax(logits).item():
                    wrong_tokens += 1

                if next_token == token_to_id["EOS"] or next_token == token_to_id["PAD"]:
                    break
        
        print(f"{wrong_tokens} / {seq_len} wrong tokens ({(wrong_tokens/seq_len)*100:.2f}%)")
        return generated

    def generate_stochastic_sequence(self, seq_len=2**14, generated=[token_to_id["BOS"]], temperature=1.0, device=None):
        if device is None:
            device = torch.get_default_device()
        
        input_seq = torch.tensor([generated], device=device)
        with torch.no_grad():
            # compute token group boundaries programmatically
            n_special = len(special_tokens)
            n_pitch = len(pitch_tokens)
            n_velocity = len(velocity_tokens)
            n_duration = len(duration_tokens)
            n_delta = len(delta_tokens)

            pitch_start = n_special
            pitch_end = pitch_start + n_pitch
            velocity_start = pitch_end
            velocity_end = velocity_start + n_velocity
            duration_start = velocity_end
            duration_end = duration_start + n_duration
            delta_start = duration_end
            delta_end = delta_start + n_delta

            # incremental stochastic generation using hidden states
            hidden_states = None
            for _ in range(seq_len - 1):
                logits, hidden_states = self.step(generated[-1], hidden_states=hidden_states, device=device)
                next_token_logits = logits / temperature
                precedent_token = id_to_token[generated[-1]]

                if precedent_token.startswith("PITCH_"):
                    allow_start, allow_end = velocity_start, velocity_end
                elif precedent_token.startswith("VELOCITY_"):
                    allow_start, allow_end = duration_start, duration_end
                elif precedent_token.startswith("DURATION_"):
                    allow_start, allow_end = delta_start, delta_end
                elif precedent_token.startswith("DELTA_"):
                    allow_start, allow_end = pitch_start, pitch_end
                elif precedent_token == "BOS":
                    allow_start, allow_end = pitch_start, pitch_end
                else:
                    break

                masked_logits = next_token_logits.clone()
                if allow_start > 0:
                    masked_logits[:allow_start] = -float('inf')
                if allow_end < masked_logits.size(0):
                    masked_logits[allow_end:] = -float('inf')

                probabilities = torch.softmax(masked_logits, dim=0).cpu().numpy()
                if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
                    break
                next_token = int(np.random.choice(len(all_tokens), p=probabilities))
                generated.append(next_token)
        return generated
    
    def quick_test(self, name):
        generated = self.generate_valid_sequence(seq_len=256+1)
        write_midi_file(generated[1:], f"generated/qt_{self.name}_{name}.mid")
        print(f"Quick test MIDI file saved as generated/qt_{self.name}_{name}.mid")

    def launch_base_training(self, dataset, epochs=10, batch_size=128, lr=0.001):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                generator=torch.Generator(device=torch.get_default_device()),
                                collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=token_to_id["PAD"]))
        criterion = nn.CrossEntropyLoss(ignore_index=token_to_id["PAD"])
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=8*lr,
            steps_per_epoch=len(dataloader),
            epochs=epochs,
            anneal_strategy='cos'
        )

        print(f"Starting training of {self.name} for {epochs} epochs on {torch.get_default_device()}...")
        logging.info(f"STARTING TRAINING OF {self.name}")
        logging.info(f"Dataset size: {len(dataset)} sequences")
        logging.info(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")
        logging.info(f"Criterion: {criterion}")
        logging.info(f"Optimizer: {optimizer}")
        logging.info(f"Scheduler: {scheduler}")
        logging.info(f"Model architecture: {self}")
        logging.info("Go !!!")

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                
                batch = batch.to(torch.get_default_device())

                targets = batch[:, 1:]  # Next token prediction
                inputs = batch[:, :-1]  # Align inputs with targets
                

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs.view(-1, len(all_tokens)), targets.contiguous().view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler._last_lr[0]:.6f}")
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler._last_lr[0]:.6f}")

            if (epoch + 1) % (epochs // 100) == 0 or epoch == epochs - 1:
                if os.path.exists(f'trainings/{self.name}') == False:
                    os.makedirs(f'trainings/{self.name}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, f'trainings/{self.name}/{self.name}_epoch{epoch+1}.pth')
        
        print(f"Training of {self.name} finished.")
        logging.info(f"Training of {self.name} finished.")
        logging.info("=========================================")
    
    def launch_lora_training(self, dataset, epochs=10, batch_size=64, lr=0.001, r=4):
        return