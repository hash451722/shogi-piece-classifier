import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



class Net(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, num_classes)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        # print(x.shape)  # torch.Size([bs, 3136])

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=1)
        return x


def load_pretrained_model(path_pt:pathlib.Path ,num_classes:int=29):
    '''
    Load pre-trained model
    Learning only at output layer
    Fix parameters except output layer
    '''
    m_state_dict = torch.load(path_pt)
    net_pre = Net(len(m_state_dict['output_layer.bias']))
    net_pre.load_state_dict(m_state_dict)

    # Fix parameters
    for param in net_pre.parameters():
        param.requires_grad = False

    # Change the number of output classes in the output layer.
    # The changed output layer automatically turns on gradient computation (learning target).
    net_pre.output_layer = nn.Linear(in_features=net_pre.output_layer.in_features, out_features=num_classes)

    # print(net_pre)
    # for name, param in net_pre.named_parameters():
    #     print('name  : ', name)
    #     print('param : ', param)
    return net_pre




def mobilenet(model:str="v3small", num_classes:int=29):
    if model == "v2":
        net = torchvision.models.mobilenet_v2()
    elif model == "v3large":
        net = torchvision.models.mobilenet_v3_large()
    else:
        net = torchvision.models.mobilenet_v3_small()

    if num_classes is not None:
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, out_features=num_classes)

    return net





def onnx_export(model, device="cpu", filename="piece.onnx", bs=1, channels=1, height=64, width=64):
    dummy_input = torch.randn(bs, channels, height, width, requires_grad=True).to(device)
    model.eval()

    torch.onnx.export(
        model,                     # model being run
        dummy_input,               # model input (or a tuple for multiple inputs)
        filename,                  # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=11,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes = {'input'  : {0 : 'batch_size'},    # variable length axes
                        'output' : {0 : 'batch_size'}})



if __name__ == '__main__':
    # model = Net()

    # data = torch.randn(81, 1, 64, 64)  # 0-1
    # output = model( data )

    # print(output)
    # print(model)

    # model.output_layer = nn.Linear(in_features=model.output_layer.in_features, out_features=2)
    # print(model)


    # for name, param in model.named_parameters():
    #     print('name  : ', name)
    #     # print('param : ', param)

    # # onnx_export(model)



    model = mobilenet(model="v3small", num_classes=29)
    data = torch.randn(81, 3, 64, 64)  # 0-1
    model.eval()
    output = model(data)
    print(model)
    # print(output)
    print(output.shape)

    print(type(model))

    onnx_export(model, filename="mobilenetv3small__.onnx", channels=3)



    # Reference
    # https://arxiv.org/abs/1905.02244
