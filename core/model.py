from torch.optim.lr_scheduler import _LRScheduler
import clip
import torch

def load_model(model_name, device):
    model, preprocess = clip.load(model_name, device=device, jit=False)  #Must set jit=False for training
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

    return model, preprocess


def get_optimizer(model, train_dataloader=None, num_epochs=None):
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(
    #     params, lr=5e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.05)
    optimizer = torch.optim.Adam(params, lr=5e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*(num_epochs-1))

    if (train_dataloader is not None) & (num_epochs is not None):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_dataloader)*num_epochs//3, T_mult=1, eta_min=1e-7, last_epoch=-1)
        return params, optimizer, scheduler
    else:
        return params, optimizer, None
    
def get_tokenize(texts):
    return clip.tokenize(texts)