import pathlib

import torch
import torchvision
from PIL import Image
import numpy as np



class ShogiPieceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:pathlib.Path) -> None:
        super().__init__()
        self.class_to_idx = None  # dict
        self.idx_to_class = None  # dict
        self.img_path_list = list(data_dir.glob('**/*.png')) + list(data_dir.glob('**/*.jpg'))
        self.labels = []
        self.imgs = []
        self.mean = None
        self.std = None
        self._preprocess(self.img_path_list)
        self.transforms = self._transforms()
        self._class_to_idx()
        self._idx_to_class()


    def _transforms(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.Normalize(self.mean , self.std)  # Standardization (Mean, Standard deviations)
        ])
        return transform


    def __getitem__(self, index:int) -> tuple[np.ndarray, str]:
        label = self.labels[index]
        img = self.imgs[index]
        img = np.array(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, self.class_to_idx[label])


    def __len__(self) -> int:
        return len(self.imgs)


    def _preprocess(self, img_path_list:list[pathlib.Path]) -> None:
        mean_sum = np.array([0, 0, 0])
        std_sum = np.array([0, 0, 0])

        for p in img_path_list:
            label = p.parent.name  # Directory name
            img = Image.open(p)
            mean, std = self._mean_std(img)

            # Empty cell
            if label == "em" or label == "emp":
                self.labels.append("emp")
                self.imgs.append(img)
                mean_sum = mean_sum + mean
                std_sum = std_sum + std
                continue

            # Black piece
            self.labels.append("b" + label)
            self.imgs.append(img)
            # White piece
            self.labels.append("w" + label)
            self.imgs.append(img.rotate(180))

            mean_sum = mean_sum + mean*2
            std_sum = std_sum + std*2

        self.mean = mean_sum / len(self.imgs)
        self.std = std_sum / len(self.imgs)


    def _mean_std(self, img_pil):
        img = np.array(img_pil)  # (64, 64, 3) (H, W, C) (0-255) (RGB)
        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]
        img_mean = [np.mean(img_r)/255, np.mean(img_g)/255, np.mean(img_b)/255]
        img_std = [np.std(img_r)/255, np.std(img_g)/255, np.std(img_b)/255]
        return np.array(img_mean), np.array(img_std)


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
    augmentation = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(degrees=(30), fill=(0.0-mean)/std),
            torchvision.transforms.RandomErasing()
        ])

    for i in range(2):
        # Load a batch of images
        imgs, _ = next(iter(dataloader))
        imgs = augmentation(imgs)

        img = torchvision.utils.make_grid(imgs)
        img_pil = torchvision.transforms.functional.to_pil_image(img)

        img_pil.save("sample_ds" + str(i) + ".png")
        # img_pil.show()



if __name__ == '__main__':
    path_current_dir = pathlib.Path(__file__).parent
    path_img_dir = path_current_dir.joinpath("images", "train_validate")

    ds = ShogiPieceDataset(path_img_dir)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    # for i, (data, target) in enumerate(loader):
    #     print(type(data))
    #     print(data.shape)
    #     break


    print(ds.class_to_idx)
    print(type(ds.class_to_idx))

    
    # _sample_images(loader, np.average(ds.mean), np.average(ds.std))

