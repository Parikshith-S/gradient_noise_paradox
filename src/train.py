import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import torchattacks
from src import config


def train_hybrid_epoch(
    model, shadow_model, optimizer, train_loader, privacy_engine, epoch
):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    # Define Attacker on the SHADOW model
    atk = torchattacks.PGD(
        shadow_model,
        eps=config.ATTACK_EPS,
        alpha=config.ATTACK_ALPHA,
        steps=config.ATTACK_STEPS,
    )

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

        # 1. SYNC: Copy weights Private -> Shadow
        # We use model._module because Opacus wraps the original model
        shadow_model.load_state_dict(model._module.state_dict())

        # 2. ATTACK: Generate adversarial examples using Shadow (No Opacus hooks)
        shadow_model.eval()
        adv_images = atk(images, labels)

        # 3. DEFEND: Train Private model on adversarial examples
        model.train()  # Opacus hooks active
        optimizer.zero_grad()

        output = model(adv_images)
        loss = criterion(output, labels)

        loss.backward()  # Opacus noise injection happens here
        optimizer.step()

        losses.append(loss.item())

        if i % 50 == 0:
            epsilon = privacy_engine.get_epsilon(config.DELTA)
            print(
                f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f} | ε: {epsilon:.2f}"
            )

    return np.mean(losses)


def evaluate(model, shadow_model, test_loader):
    # Use shadow model for evaluation to avoid Opacus overhead
    shadow_model.load_state_dict(model._module.state_dict())
    shadow_model.eval()

    correct_clean, correct_adv, total = 0, 0, 0
    atk = torchattacks.PGD(
        shadow_model, eps=config.ATTACK_EPS, alpha=config.ATTACK_ALPHA, steps=10
    )

    for images, labels in test_loader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

        # Clean
        with torch.no_grad():
            outputs = shadow_model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct_clean += (predicted == labels).sum().item()

        # Robust (Adversarial)
        adv_images = atk(images, labels)
        with torch.no_grad():
            outputs_adv = shadow_model(adv_images)
            _, predicted_adv = torch.max(outputs_adv.data, 1)
            correct_adv += (predicted_adv == labels).sum().item()

        total += labels.size(0)
        if total > 500:
            break  # Speed up for demo

    return 100 * correct_clean / total, 100 * correct_adv / total
