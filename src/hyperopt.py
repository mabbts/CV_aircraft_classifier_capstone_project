

# Model selector
def get_model(backbone_name, num_classes, dropout_rate):
    if backbone_name == "ResNet50_CAP":
        return CAPResNet(num_classes=num_classes, drop=dropout_rate)
    elif backbone_name == "EffNet_SE":
        return SEEffNet(num_classes=num_classes, drop=dropout_rate)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

# Objective function
def objective(trial):
    # Hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    backbone_name = trial.suggest_categorical('backbone', ['ResNet50_CAP', 'EffNet_SE'])
    scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])
    criterion_name = trial.suggest_categorical('criterion', ['CrossEntropy', 'LabelSmoothing', 'Focal'])

    # Data
    train_loader, val_loader, test_loader, num_classes, class_names = get_loaders(img_size=224, batch_size=batch_size, annot='variant')

    # Model
    model = get_model(backbone_name, num_classes, dropout_rate)
    model.to(device)

    # Optimizer
    optimizer_cls = getattr(optim, optimizer_name)
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # Criterion 
    if criterion_name == 'CrossEntropy':
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif criterion_name == 'LabelSmoothing':
        smoothing = trial.suggest_float('smoothing', 0.05, 0.2)
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    elif criterion_name == 'Focal':
        gamma = trial.suggest_float('gamma', 1.0, 3.0)
        criterion = FocalLoss(gamma=gamma)

    scaler = GradScaler('cuda')

    # Training loop
    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss, train_acc, _, _ = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print("✅ Model improved. Saving...")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s).")
        if epochs_without_improvement >= patience:
            print("⏹️ Early stopping triggered.")
            break

    return val_acc
