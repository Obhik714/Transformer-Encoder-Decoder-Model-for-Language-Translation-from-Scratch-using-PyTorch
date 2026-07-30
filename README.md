# Custom Encoder-Decoder Model

A complete, from-scratch implementation of a sequence-to-sequence Encoder-Decoder architecture using PyTorch. This project leverages Hugging Face's `AutoTokenizer` for efficient and robust text tokenization.

## 🚀 Features

- **Custom PyTorch Architecture**: Full implementation of the Encoder and Decoder modules from the ground up.
- **Hugging Face Integration**: Uses `AutoTokenizer` to cleanly handle vocabulary building, padding, and token management.
- **Modular Design**: Clean separation of dataset processing, model architecture, training loops, and evaluation scripts.
- **State Dict Management**: Easily save and load isolated model weights (`.pth` files) for inference.

## 📂 Project Structure

```text
├── data/                  # Raw and processed datasets
├── models/                # Saved weights (e.g., encoder.pth, decoder.pth)
├── reports/               # Training metrics and figures (loss plots)
├── src/                   # Source code for the project
│   ├── dataset.py         # Data loading and Tokenizer setup
│   ├── architecture.py    # PyTorch Encoder and Decoder classes
│   ├── plots.py           # Visualization and monitoring scripts
│   └── predict.py         # Inference and evaluation routines
├── pyproject.toml         # Project dependencies and environment configuration
└── README.md              # Project documentation
```

## 🧠 Usage

### 1. Tokenization
The project uses `AutoTokenizer` to seamlessly convert raw text into model-ready tensors:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
tokens = tokenizer("Example sequence for the encoder", return_tensors="pt")
```

### 2. Loading the Model for Inference
Load your trained state dictionaries into the custom architecture:

```python
import torch
from src.architecture import Encoder, Decoder

# Initialize the models
encoder = Encoder(...) # Add your hyperparameters
decoder = Decoder(...)

# Load the saved state_dicts
encoder.load_state_dict(torch.load('models/encoder.pth'))
decoder.load_state_dict(torch.load('models/decoder.pth'))

# Set to evaluation mode
encoder.eval()
decoder.eval()
```

### 3. Monitoring Training
To generate loss visualizations (training vs. validation) and save them to the reports folder, execute the plotting module:
```bash
python -m src.plots
```

## 📈 Results
Metrics and loss visualizations (such as `training_vs_validation_losses.png`) are automatically saved to `reports/figures/` during the training cycle to help monitor convergence and prevent overfitting.