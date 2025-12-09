# gradient_noise_paradox
Investigating the "Gradient Noise Paradox" in AI Safety: A study on the conflict between Differential Privacy (DP-SGD) and Adversarial Training. Uses a custom "Shadow Model" pipeline to synchronize Opacus with PGD attacks, demonstrating how privacy-preserving noise systematically degrades model robustness. #Robustness #AISafety

# The Gradient Noise Paradox: Robustness vs. Privacy

## Project Overview
This project investigates a critical failure mode in AI Safety: the conflict between **Differential Privacy (DP)** and **Adversarial Robustness**. 

**Hypothesis:** The gradient clipping and noise injection required for privacy destroys the fine-grained decision boundaries required for adversarial robustness.

## The Architecture
We utilize a **Shadow Model** technique to overcome the incompatibility between Opacus (Privacy Library) and TorchAttacks (Adversarial Library).
1. **Private Model:** Wrapped with `Opacus` for DP-SGD.
2. **Shadow Model:** A synced copy used to generate PGD attacks without triggering privacy hooks.

## Installation
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate