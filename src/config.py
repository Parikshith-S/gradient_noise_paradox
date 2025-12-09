import torch

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 10

# Privacy (The knobs you turn)
SIGMA = 1.0       # Noise Multiplier (Higher = More Privacy / Less Robustness)
MAX_GRAD_NORM = 1.2
DELTA = 1e-5

# Robustness
ATTACK_EPS = 8/255
ATTACK_ALPHA = 2/255
ATTACK_STEPS = 4