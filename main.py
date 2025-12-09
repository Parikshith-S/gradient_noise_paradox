import torch.optim as optim
from opacus import PrivacyEngine
from src import config, models, utils, train

def main():
    print(f"Running on {config.DEVICE}")
    
    # 1. Setup Data
    train_loader, test_loader = utils.get_dataloaders(config.BATCH_SIZE)
    
    # 2. Setup Models
    # Main model (will be wrapped)
    model = models.get_safe_model(device=config.DEVICE)
    # Shadow model (never wrapped, used for attack gen)
    shadow_model = models.get_safe_model(device=config.DEVICE)
    
    optimizer = optim.RMSprop(model.parameters(), lr=config.LR)
    
    # 3. Setup Privacy
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=config.SIGMA,
        max_grad_norm=config.MAX_GRAD_NORM,
    )
    
    # 4. Loop
    history = {'clean': [], 'robust': [], 'epsilon': []}
    
    for epoch in range(1, config.EPOCHS + 1):
        loss = train.train_hybrid_epoch(model, shadow_model, optimizer, train_loader, privacy_engine, epoch)
        acc_clean, acc_adv = train.evaluate(model, shadow_model, test_loader)
        eps = privacy_engine.get_epsilon(config.DELTA)
        
        history['clean'].append(acc_clean)
        history['robust'].append(acc_adv)
        history['epsilon'].append(eps)
        
        print(f"End Epoch {epoch}: Clean: {acc_clean:.2f}% | Robust: {acc_adv:.2f}% | Eps: {eps:.2f}")

    # 5. Save
    utils.plot_results(history)

if __name__ == "__main__":
    main()