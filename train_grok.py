import copy
import einops
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cosine_similarity
import tqdm.auto as tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig
import wandb


def make_dataset(p: int, device):
    # For p**4 elements add a d vector with l -> (i j k l)
    a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
    b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
    equals_vector = einops.repeat(torch.tensor(p), " -> (i j)", i=p, j=p)
    plus_vector = einops.repeat(torch.tensor(p+1), " -> (i j)", i=p, j=p)

    dataset = torch.stack([
        a_vector,
        plus_vector,
        b_vector,
        equals_vector], dim=1).to(device)
    labels = ((dataset[:, 0] + dataset[:, 2])) % p
    return dataset, labels


def get_dataloaders(p: int, frac_train: float, batch_size: int, device):
    dataset, labels = make_dataset(p, device)
    indices = torch.randperm(p**2)
    cutoff = int((p**2) * frac_train)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]
    train_data = TensorDataset(dataset[train_indices], labels[train_indices])
    test_data = TensorDataset(dataset[test_indices], labels[test_indices])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


def loss_fn(logits, labels):
    if len(logits.shape)==3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()


def train_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda', requires_grad=False)
    for batch, labels in dataloader:
        logits = model(batch)
        loss = loss_fn(logits, labels)
        loss.backward()
        total_loss += loss
    return total_loss


def test_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda', requires_grad=False)
    for batch, labels in dataloader:
        logits = model(batch)
        loss = loss_fn(logits, labels)
        total_loss += loss
    return total_loss

    
def get_grads(model):

    return {
        name: copy.deepcopy(param.grad).ravel() for name, param in model.named_parameters() if param.requires_grad
    }


def grad_similarity(train_grads, test_grads):
    full_grads = {
        name: (train_grads[name] + test_grads[name])/2 for name in train_grads
    }

    similarity = {
        name: torch.dot(train_grads[name], full_grads[name]) for name in train_grads
    }
    return similarity, full_grads


def train(model, optimizer, train_dataloader, test_dataloader, checkpoint_every, num_epochs, grok_threshold):
    train_losses = []
    test_losses = []
    model_checkpoints = []
    opt_checkpoints = []
    checkpoint_epochs = []

    for epoch in tqdm.tqdm(range(num_epochs)):
        train_loss = train_forward(model, train_dataloader)
        np_train = train_loss.item()
        train_losses.append(np_train)

        train_grads = get_grads(model)

        optimizer.step()
        optimizer.zero_grad()

        
        test_loss = test_forward(model, test_dataloader)
        np_test = test_loss.item()
        test_losses.append(np_test)

        test_grads = get_grads(model)
        optimizer.zero_grad()
        for param in model.parameters():
            if param.requires_grad:
                param.grad = None

        similarity, full_grads = grad_similarity(train_grads, test_grads)

        similarity = {f'gradient_alignment/{n}': sim for n, sim in similarity.items()}
        
        msg = {'loss/train': np_train, 'loss/test': np_test}
        msg.update(similarity)


        if (epoch % 100) == 0:
            train_grads = {f'train_gradients/{n}': g for n, g in train_grads.items()}
            full_grads = {f'full_gradients/{n}': g for n, g in full_grads.items()}
            test_grads = {f'test_gradients/n': g for n, g in test_grads.items()}
            msg.update(train_grads)
            msg.update(full_grads)
            msg.update(test_grads)

        wandb.log(msg)

        if (epoch % checkpoint_every) == 0:
            checkpoint_epochs.append(epoch)
            model_state = copy.deepcopy(model.state_dict())
            opt_state = copy.deepcopy(optimizer.state_dict())
            torch.save(
                {
                    "model": model_state,
                    "optimizer": opt_state,
                    "config": model.cfg,
                    "rng": torch.get_rng_state()
                },
                f'checkpoints/xy33/{epoch}.pth'
            )
            model_checkpoints.append(model_state)
            opt_checkpoints.append(opt_state)
            #print(f"Epoch {epoch} Train Loss {np_train} Test Loss {np_test}")

        if test_loss.item() <= grok_threshold:
            break
    torch.save(
     {
         "model":model.state_dict(),
         "config": model.cfg,
         "checkpoints": model_checkpoints,
         "checkpoint_epochs": checkpoint_epochs,
         "test_losses": test_losses,
         "train_losses": train_losses
     },
     "grokking_xy_33_full_run.pth")

def main():
    p = 113
    frac_train = 1.0 / 3
    lr = 1e-3
    wd = 1. 
    betas = (0.9, 0.98)
    num_epochs = 50_000
    grok_threshold = 0.00001
    checkpoint_every = 100
    batch_size = 2 ** 16
    device = 'cuda'
    seed = 999

    cfg = HookedTransformerConfig(
        n_layers = 1,
        n_heads = 4,
        d_model = 128,
        d_head = 32,
        d_mlp = 512,
        act_fn = "relu",
        normalization_type=None,
        d_vocab=p+2,
        d_vocab_out=p,
        n_ctx=4,
        init_weights=True,
        device=device,
        seed = seed,
    )
    model = HookedTransformer(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=betas
    )

    wandb.init(
        entity="dstander",
        project="grokking_galois",
        group="x+y",
        config=cfg.to_dict()
    )

    #wandb.watch(model, log_freq=100)

    train_data, test_data = get_dataloaders(p, frac_train, batch_size, device)

    train(
        model,
        optimizer,
        train_data,
        test_data,
        checkpoint_every,
        num_epochs,
        grok_threshold
    )


if __name__ == '__main__':
    main()

