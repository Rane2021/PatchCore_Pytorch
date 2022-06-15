import torch
import torchvision.models as models

# TODO: CUDA version
# vgg16 = models.resnet18(pretrained=True)
# vgg16.cuda()
# vgg16.eval()

# example = torch.rand(1, 3, 224, 224).cuda()
# traced_model = torch.jit.trace(vgg16, example)
# traced_model.save("models/resnet18_trace.pt")
# print("traced model gen finish!")

# output = traced_model(torch.ones(1, 3, 224, 224).cuda())
# print(output.max())

# scripted_model = torch.jit.script(vgg16)
# scripted_model.save("models/resnet18_script.pt")
# print("scripted model gen finish!")


# TODO: CPU version
vgg16 = models.resnet18(pretrained=True)
vgg16.eval()

example = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(vgg16, example)
traced_model.save("../models/resnet18_trace.pt")
print("traced model gen finish!")

output = traced_model(torch.ones(1, 3, 224, 224))
print(output.max())  # tensor(4.4229, grad_fn=<MaxBackward1>)

scripted_model = torch.jit.script(vgg16)
scripted_model.save("../models/resnet18_script.pt")
print("scripted model gen finish!")


