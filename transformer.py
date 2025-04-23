'''
This is our transformer model implementation.
This script reads in the MIDI tensor data, defines a transformer model, trains it, and saves the model.
'''

# Note to Alex: use the Conda 'myenv' environment
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path


training_data = []
training_labels = []
test_data = []
test_labels = []
validation_data = []
validation_labels = []


def load_midi_data():
    """
    Load all MIDI tensors into their respective list of tensors along with their labels. 
    """
    # Location of the MIDI tensor data
    base_dir="dataset/maestro-v2.0.0-tensors"
    metadata_file="dataset/maestro-v2.0.0.csv"

    # Read the metadata file
    with open(metadata_file, 'r') as f:
        lines = f.readlines()

    # Skip the header line
    lines = lines[1:]

    # Iterate through each line in the metadata file
    for line in lines:
        # Split the line into components
        parts = line.strip().split(',')

        # Get the tensor file name and label
        tensor_file_name = (parts[4] + ".pt").strip()   # Tensor file name is just the MIDI file name with .pt extension appended
        label = parts[0].strip()                        # Label (Composer)
        split = parts[2].strip()                        # Split (train/test/validation)

        # Load the tensor file
        tensor_path = os.path.join(base_dir, tensor_file_name)

        if os.path.exists(tensor_path):            
            try:
                tensor = torch.load(tensor_path, weights_only=True)  # Need the 'weights_only' argument to avoid a warning, maybe try without it if you are getting an error

                # Append the tensor and label to the respective lists only if loading is successful
                if split == "train":
                    training_data.append(tensor)
                    training_labels.append(label)
                elif split == "test":
                    test_data.append(tensor)
                    test_labels.append(label)
                elif split == "validation":
                    validation_data.append(tensor)
                    validation_labels.append(label)
                else:
                    print(f"ERROR: Unknown split {split}. Skipping...")
            except Exception as e:
                print(f"ERROR: Could not load tensor from {tensor_path}: {e}")

    # Print the number of tensors loaded
    print(f"Loaded {len(training_data)} training tensors, {len(test_data)} test tensors, and {len(validation_data)} validation tensors.")


class MIDITransformer(nn.Module):
    """
    Transformer model for MIDI data.
    """
    def __init__(self):
        super(MIDITransformer, self).__init__()

        # Model Architecture
        #TODO: Define the transformer model architecture here
    
    def forward(self, x):
        # Forward pass
        #TODO: Define the forward pass here
        pass


if __name__ == "__main__":
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device") #! If this prints "Using cpu device", then you are not using the GPU

    # Load the MIDI tensor data
    # Note that the dataset is only 2.4MB, so it should fit on any modern GPU
    load_midi_data()

    # Define the model
    model = MIDITransformer().to(device)