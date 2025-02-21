from music21 import stream, note, meter, instrument, midi
import random
import os
import subprocess
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import json

# MuseScore path (global variable, adjust if needed)
musescore_path = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

def create_score(output_dir, base_name="score", num_measures=4):
    """
    Generate a random guitar score and save it as MusicXML, MIDI, and a tablature image.
    
    Args:
        output_dir (str): Directory to save the files.
        base_name (str): Base name for output files (e.g., 'score' -> 'score.xml', 'score.mid', 'score.png').
        num_measures (int): Number of measures in the score.
    
    Returns:
        tuple: Paths to the MusicXML, MIDI, and image files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create the score
    score = stream.Score()
    part = stream.Part()
    part.insert(0, instrument.Guitar())
    score.insert(0, meter.TimeSignature('4/4'))

    for _ in range(num_measures):
        measure = stream.Measure()
        remaining_beats = 4.0
        while remaining_beats > 0:
            duration = random.choice([0.5, 1.0])  # eighth or quarter note
            if duration > remaining_beats:
                duration = remaining_beats
            string = random.randint(1, 6)
            fret = random.randint(0, 12)
            n = note.Note()
            n.pitch.midi = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}[string] + fret
            n.quarterLength = duration
            measure.append(n)
            remaining_beats -= duration
        part.append(measure)
    score.append(part)

    # Define output paths
    musicxml_path = os.path.join(output_dir, f"{base_name}.xml")
    midi_path = os.path.join(output_dir, f"{base_name}.mid")
    image_base_path = os.path.join(output_dir, f"{base_name}.png")

    # Remove any existing files with .png to prevent MuseScore from appending suffixes
    for fname in os.listdir(output_dir):
        if fname.startswith(base_name) and fname.endswith('.png'):
            os.remove(os.path.join(output_dir, fname))

    # Save as MusicXML
    score.write('musicxml', fp=musicxml_path)
    print(f"MusicXML saved at: {musicxml_path}")

    # Save as MIDI
    score.write('midi', fp=midi_path)
    print(f"MIDI saved at: {midi_path}")

    # Convert MusicXML to PNG with tablature using MuseScore
    try:
        # Use a simpler command without shell=True, ensuring proper path formatting
        cmd = [musescore_path, "-o", image_base_path, "-T", "0", musicxml_path]
        print(f"Running MuseScore command: {' '.join(cmd)}")  # Debug output
        subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(f"Tablature image saved at: {image_base_path}")

        # Check if MuseScore added a .1 suffix and rename if necessary
        image_files = [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith('.png')]
        if image_files:
            actual_image_path = os.path.join(output_dir, image_files[0])
            if actual_image_path != image_base_path:
                os.rename(actual_image_path, image_base_path)
                print(f"Renamed image to: {image_base_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running MuseScore: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"MuseScore not found at {musescore_path}. Please check the path.")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return musicxml_path, midi_path, image_base_path

def midi_to_tokens(midi_path):
    """
    Convert a MIDI file to a sequence of tokens.
    
    Args:
        midi_path (str): Path to the MIDI file.
    
    Returns:
        list: Sequence of tokens (e.g., ['START', 'NOTE_ON_64', ...]).
    """
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    events = []
    for track in mf.tracks:
        time = 0
        for event in track.events:
            if isinstance(event, midi.DeltaTime):
                time += event.time
            elif event.isNoteOn() and event.velocity > 0:
                events.append((time, f'NOTE_ON_{event.pitch}'))
            elif event.isNoteOff() or (event.isNoteOn() and event.velocity == 0):
                events.append((time, f'NOTE_OFF_{event.pitch}'))
    events.sort()
    token_sequence = ['START']
    prev_time = 0
    for time, evt in events:
        if time > prev_time:
            token_sequence.append(f'TIME_SHIFT_{time - prev_time}')
        token_sequence.append(evt)
        prev_time = time
    token_sequence.append('END')
    return token_sequence

def clear_output_dir(output_dir):
    """
    Clear out existing files in the output directory with specified extensions.
    
    Args:
        output_dir (str): Directory to clear.
    """
    if os.path.exists(output_dir):
        for fname in os.listdir(output_dir):
            if fname.endswith(('.xml', '.mid', '.png')):
                file_path = os.path.join(output_dir, fname)
                os.remove(file_path)
                print(f"Removed existing file: {file_path}")

def build_vocabulary(token_sequences):
    """
    Build a vocabulary mapping token strings to integer IDs.
    
    Args:
        token_sequences (list of lists): List of token sequences from MIDI files.
    
    Returns:
        tuple: (token_to_id, id_to_token) dictionaries.
    """
    # Collect all unique tokens
    unique_tokens = set()
    for sequence in token_sequences:
        unique_tokens.update(sequence)

    # Create mappings
    token_to_id = {}
    id_to_token = {}
    current_id = 0

    # Add special tokens first (if not already present)
    for special_token in ['START', 'END']:
        if special_token not in token_to_id:
            token_to_id[special_token] = current_id
            id_to_token[current_id] = special_token
            current_id += 1

    # Add other tokens
    for token in sorted(unique_tokens):
        if token not in token_to_id:
            token_to_id[token] = current_id
            id_to_token[current_id] = token
            current_id += 1

    return token_to_id, id_to_token

def preprocess_image(image_path, target_size=(128, 1024)):
    """
    Preprocess a tablature image into a tensor.
    
    Args:
        image_path (str): Path to the PNG image.
        target_size (tuple): Target size for the image (height, width).
    
    Returns:
        torch.Tensor: Preprocessed image tensor, normalized to [-1, 1].
    """
    transform = T.Compose([
        T.Resize(target_size),  # Resize to fixed height and width
        T.ToTensor(),  # Convert to tensor [0, 1]
        T.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    img = Image.open(image_path).convert('L')  # Grayscale
    img_tensor = transform(img)
    return img_tensor

def tokens_to_ids(token_sequence, token_to_id):
    """
    Convert a token sequence to integer IDs using the vocabulary.
    
    Args:
        token_sequence (list): List of token strings.
        token_to_id (dict): Mapping of tokens to integer IDs.
    
    Returns:
        list: List of integer IDs.
    """
    return [token_to_id[token] for token in token_sequence]

class TabMidiDataset(Dataset):
    """
    Custom PyTorch Dataset for tablature images and MIDI token sequences.
    """
    def __init__(self, image_paths, token_sequences, token_to_id):
        """
        Initialize the dataset.
        
        Args:
            image_paths (list): List of paths to tablature images.
            token_sequences (list): List of token sequences (strings).
            token_to_id (dict): Mapping of tokens to integer IDs.
        """
        self.image_paths = image_paths
        self.token_sequences = token_sequences
        self.token_to_id = token_to_id

    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a sample (image tensor, integer token sequence) at the given index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            tuple: (image_tensor, token_ids), where token_ids are integers.
        """
        # Preprocess image
        image_path = self.image_paths[idx]
        image_tensor = preprocess_image(image_path)

        # Convert tokens to IDs
        token_sequence = self.token_sequences[idx]
        token_ids = tokens_to_ids(token_sequence, self.token_to_id)

        return image_tensor, torch.tensor(token_ids, dtype=torch.long)

# ---------
# PARAMS
training_dir = r"D:\MusicParser\TrainingData"
num_samples = 5  # Generate 5 samples for testing

generateTrainingData = False
trainingRun = True
# ---------

if generateTrainingData:
    # Clear existing training data
    clear_output_dir(training_dir)

    # Generate training data and save token sequences and image paths
    all_token_sequences = []
    image_paths = []

    for i in range(num_samples):
        print(f"\nGenerating sample {i + 1}...")
        # Generate score with unique file names
        _, midi_path, image_path = create_score(
            output_dir=training_dir,
            base_name=f"sample_{i}",
            num_measures=4
        )
        image_paths.append(image_path)
        # Convert MIDI to tokens
        tokens = midi_to_tokens(midi_path)
        print(f"Token sequence for sample {i}: {tokens}")
        all_token_sequences.append(tokens)

    # Build and save the vocabulary
    token_to_id, id_to_token = build_vocabulary(all_token_sequences)
    print("\nToken to ID mapping:")
    for token, idx in token_to_id.items():
        print(f"{token}: {idx}")
    print("\nID to Token mapping:")
    for idx, token in id_to_token.items():
        print(f"{idx}: {token}")

    # Save token sequences and image paths to JSON files
    data_path = os.path.join(training_dir, "training_data.json")
    with open(data_path, 'w') as f:
        json.dump({"image_paths": image_paths, "token_sequences": all_token_sequences}, f)
    print(f"Training data saved at: {data_path}")

    # Save the vocabulary to a file for later use
    vocab_path = os.path.join(training_dir, "vocabulary.json")
    with open(vocab_path, 'w') as f:
        json.dump({"token_to_id": token_to_id, "id_to_token": id_to_token}, f)
    print(f"Vocabulary saved at: {vocab_path}")

if trainingRun:
    # Load training data and vocabulary for training
    data_path = os.path.join(training_dir, "training_data.json")
    vocab_path = os.path.join(training_dir, "vocabulary.json")

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    token_to_id = vocab["token_to_id"]
    id_to_token = vocab["id_to_token"]

    # Load training data (image paths and token sequences)
    with open(data_path, 'r') as f:
        data = json.load(f)
    image_paths = data["image_paths"]
    token_sequences = data["token_sequences"]

    # Create dataset
    dataset = TabMidiDataset(image_paths, token_sequences, token_to_id)

    # Example: Create a DataLoader for batching
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Example: Iterate over the dataset to verify
    for images, token_ids in dataloader:
        print(f"Batch of images shape: {images.shape}")  # e.g., [batch_size, 1, 128, 1024]
        print(f"Batch of token IDs shape: {token_ids.shape}")  # e.g., [batch_size, seq_len]
        print(f"Sample token IDs: {token_ids[0].tolist()}")  # First sample in batch
        break  # Just show one batch for now