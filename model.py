import pathlib

import torch
import torchvision



def network(model:str="mobilenet_v3_large", num_classes:int=29):
    if model == "mobilenet_v2":
        net = torchvision.models.mobilenet_v2()
    elif model == "mobilenet_v3_small":
        net = torchvision.models.mobilenet_v3_small()
    elif model == "mobilenet_v3_large":
        net = torchvision.models.mobilenet_v3_large()
    else:
        return None
    
    if num_classes is not None:
        net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, out_features=num_classes)

    return net



def onnx_export(model, filename:str, bs:int=1, channels:int=3, height:int=64, width:int=64) -> None:
    dummy_input = torch.randn(bs, channels, height, width, requires_grad=True).to("cpu")
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
    dummy_data = torch.randn(81, 3, 64, 64)  # 0-1

    model = network(model="mobilenet_v3_large", num_classes=29)
    model.eval()
    output = model(dummy_data)

    print(model)
    print(type(model))

    print(output)
    print(output.shape)

    # onnx_export(model, filename="mobilenet_v3_large.onnx")

    # for name, param in model.named_parameters():
    #     print('name  : ', name)
    #     # print('param : ', param)

