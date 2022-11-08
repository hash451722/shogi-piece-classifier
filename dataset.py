import pathlib
import pickle

import torch
import torchvision
from PIL import Image
import numpy as np



class ShogiPieceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:pathlib.Path, transforms=None) -> None:
        super().__init__()
        self.class_to_idx = None  # dict
        self.idx_to_class = None  # dict
        self.transforms = transforms
        self.img_path_list = list(data_dir.glob('**/*.png')) + list(data_dir.glob('**/*.jpg'))
        self.labels = []
        self.imgs = []
        self._rotate(self.img_path_list)
        self._class_to_idx()
        self._idx_to_class()


    def __getitem__(self, index:int) -> tuple[np.ndarray, str]:
        label = self.labels[index]
        img = self.imgs[index]
        img = np.array(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, self.class_to_idx[label])


    def __len__(self) -> int:
        # return len(self.img_path_list)
        return len(self.imgs)


    def _rotate(self, img_path_list:list[pathlib.Path]) -> None:
        for p in img_path_list:
            label = p.parent.name  # Directory name
            img = Image.open(p)
            # Empty cell
            if label == "em" or label == "emp":
                self.labels.append("emp")
                self.imgs.append(img)
                continue
            # Black piece
            self.labels.append("b" + label)
            self.imgs.append(img)
            # White piece
            self.labels.append("w" + label)
            self.imgs.append(img.rotate(180))


    def _class_to_idx(self) -> None:
        cls = self.labels[:]  # deep copy
        cls = list(set(cls))
        cls.sort()
        idx = list( range( len(cls) ) )
        self.class_to_idx = dict(zip(cls, idx))


    def _idx_to_class(self) -> None:
        d = {}
        for k, v in self.class_to_idx.items():
            d[str(v)] = k
        self.idx_to_class = d




def _sample_images(dataloader, mean:float=0.0, std:float=1.0):
    transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(degrees=(30), fill=(0.0-mean)/std),
            torchvision.transforms.RandomErasing()
        ])

    for i in range(2):

        # Load a batch of images
        imgs, _ = next(iter(dataloader))
        imgs = transform(imgs)


        img = torchvision.utils.make_grid(imgs)
        img_pil = torchvision.transforms.functional.to_pil_image(img)

        img_pil.save("sample_ds" + str(i) + ".png")
        # img_pil.show()


def statistics(path_img_dir:pathlib.Path, save:bool=True) -> dict:
    transform_basic = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((64, 64)),
        ])

    ds = ShogiPieceDataset(path_img_dir, transforms=transform_basic)
    num_class = ds.class_to_idx.copy()
    num_class = {k: 0 for k in num_class.keys()}

    mean_sum = 0
    std_sum = 0
    for (img, idx) in ds:
        cls = ds.idx_to_class[str(idx)]
        num_class[cls] += 1

        mean_sum += torch.mean(img)
        std_sum += torch.std(img)

    mean = mean_sum / len(ds)
    std = std_sum / len(ds)


    stats = {
        "mean":mean.to('cpu').detach().numpy().copy().item(),
        "std":std.to('cpu').detach().numpy().copy().item(),
        "num_data":len(ds),
        "num_class":num_class
        }

    if save:
        with open(path_img_dir.joinpath("stats.pickle"), 'wb') as f:
            pickle.dump(stats, f)

    return stats





if __name__ == '__main__':
    path_current_dir = pathlib.Path(__file__).parent
    path_img = path_current_dir.joinpath("train_images_original")


    _stats = statistics(path_img)
    # print(_stats)

    with open(path_img.joinpath("stats.pickle"), mode="rb") as f:
        stats = pickle.load(f)

    print(stats)


    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((64, 64)),
            # torchvision.transforms.Normalize((stats["mean"]), (stats["std"]))  # Standardization (Mean, Standard deviations)
        ])

    ds = ShogiPieceDataset(path_img, transforms=transform)


    print(ds)
    print(type(ds))

    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    # for i, (data, target) in enumerate(loader):
    #     print(type(data))
    #     print(data.shape)
    #     break
    
    _sample_images(loader)

