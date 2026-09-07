import torch

from losses import HybridFocalFedProxLoss

torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available(),
)


def train_local(
    model,
    trainloader,
    epochs=2,
    lr=0.001,
    mu=0.001,
):

    model = model.to(DEVICE)

    global_params = [
        p.detach().clone()
        for p in model.parameters()
    ]

    criterion = HybridFocalFedProxLoss(
        alpha=0.80,
        gamma=2.0,
        mu=mu,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    model.train()

    for _ in range(epochs):

        for x, y in trainloader:

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available(),
            ):

                outputs = model(x)

                loss = criterion(
                    outputs,
                    y,
                    model,
                    global_params,
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            scaler.step(optimizer)

            scaler.update()

    return model