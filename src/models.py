import torch.nn as nn
from torchvision import models
import torchvision.models.resnet as resnet_module
from opacus.validators import ModuleValidator

def patch_resnet_mechanics():
    """
    Patches torchvision's BasicBlock to avoid in-place addition (out += identity).
    This allows Opacus to calculate gradients correctly.
    """
    def safe_basic_block_forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity  # <--- The Critical Fix (Not in-place)
        out = self.relu(out)
        return out

    resnet_module.BasicBlock.forward = safe_basic_block_forward

def get_safe_model(num_classes=10, device='cpu'):
    # 1. Apply patch
    patch_resnet_mechanics()
    
    # 2. Load Standard ResNet
    model = models.resnet18(num_classes=num_classes)
    
    # 3. Replace BatchNorm with GroupNorm (Opacus requirement)
    model = ModuleValidator.fix(model)
    
    # 4. Disable inplace ReLUs
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
            
    return model.to(device)