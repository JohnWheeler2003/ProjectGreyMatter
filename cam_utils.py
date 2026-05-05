import torch
import numpy as np
import torch.nn.functional as F

def vit_reshape_transform(tensor, height=14, width=14):
    """
    Reformats Vision Transformer sequence tokens into a 2D spatial grid.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class AttentionRollout:
    def __init__(self, model, head_fusion="mean", discard_ratio=0.9, **kwargs):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []
        self.hooks = []

        for module in self.model.modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                self.hooks.append(module.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
                self.hooks.append(module.register_forward_hook(self._forward_hook))

    def _pre_hook(self, module, args, kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False 
        return args, kwargs

    def _forward_hook(self, module, args, output):
        self.attentions.append(output[1].detach().cpu())

    def __call__(self, input_tensor, targets=None):
        self.attentions.clear()
        
        with torch.no_grad():
            self.model(input_tensor)

        batch_size = input_tensor.size(0)
        rollouts = []
        
        for b in range(batch_size):
            seq_len = self.attentions[0].size(-1)
            result = torch.eye(seq_len) 
            
            for attention in self.attentions:
                attn = attention[b] 
                
                if self.head_fusion == "mean":
                    attn = attn.mean(dim=0)
                elif self.head_fusion == "max":
                    attn = attn.max(dim=0)[0]
                elif self.head_fusion == "min":
                    attn = attn.min(dim=0)[0]
                
                flat = attn.flatten()
                k = int(flat.size(0) * self.discard_ratio)
                if k > 0:
                    val, _ = torch.kthvalue(flat, k)
                    attn[attn < val] = 0
                
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
                attn = attn + torch.eye(seq_len)
                attn = attn / attn.sum(dim=-1, keepdim=True)
                
                result = torch.matmul(attn, result)
            
            mask = result[0, 1:]
            width = int(np.sqrt(mask.size(0)))
            mask = mask.reshape(1, 1, width, width).float()
            
            mask = F.interpolate(
                mask, 
                size=(input_tensor.size(2), input_tensor.size(3)), 
                mode='bilinear', 
                align_corners=False
            )
            mask = mask.squeeze().numpy()
            
            mask = mask - np.min(mask)
            mask = mask / (np.max(mask) + 1e-9)
            
            rollouts.append(mask)
            
        return np.stack(rollouts)