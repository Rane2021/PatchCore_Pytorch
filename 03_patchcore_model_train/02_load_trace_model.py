import torch
import torchvision.models as models

# TODO: CPU version
traced_model = torch.jit.load("results/patchcore/mvtec/bottle/patchcore_trace.pt")

output = traced_model(torch.ones(1, 3, 224, 224))
print(output)  
print("finish!")


