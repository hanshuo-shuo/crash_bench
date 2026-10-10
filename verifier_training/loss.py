"""Causal assistant-only next-token loss with auditable rationale/verdict masks."""
import torch
import torch.nn.functional as F


def masked_losses(logits, labels, rationale_mask, verdict_mask):
    # Prediction at t consumes only tokens through t and supervises token t+1.
    shifted = labels[:, 1:]
    token = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                            shifted.reshape(-1), ignore_index=-100, reduction='none').reshape_as(shifted)
    valid = shifted.ne(-100)
    result = {'total': token.sum()/valid.sum().clamp_min(1)}
    for name, mask in [('rationale', rationale_mask[:, 1:]), ('verdict', verdict_mask[:, 1:])]:
        mask = mask & valid
        result[name] = token[mask].mean() if mask.any() else token.sum()*0
        result[name+'_tokens'] = int(mask.sum())
    return result


def gradient_probe(model, losses, variant):
    """Prove each loss separately reaches LM LoRA, with no visual gradients."""
    params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    out = {}
    for name in (('rationale','verdict') if variant=='B' else ('verdict',)):
        grads = torch.autograd.grad(losses[name], [p for _,p in params], retain_graph=True, allow_unused=True)
        squared = sum(g.detach().float().square().sum().item() for g in grads if g is not None)
        if not (losses[name].item() > 0 and squared > 0 and torch.isfinite(torch.tensor(squared))):
            raise RuntimeError(name+' must have finite nonzero loss and LoRA gradient')
        out[name] = {'loss':losses[name].item(), 'tokens':losses[name+'_tokens'],
                     'lora_gradient_norm':squared**.5}
    if any(p.requires_grad or p.grad is not None for n,p in model.named_parameters() if 'visual' in n):
        raise RuntimeError('Vision encoder/projector unfrozen')
    return out
