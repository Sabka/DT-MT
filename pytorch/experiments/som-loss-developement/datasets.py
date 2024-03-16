from torch.utils.data import DataLoader
import torch
import numpy as np
from ucimlrepo import fetch_ucirepo
import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("ucimlrepo")  # install wine dataset repo
install("seaborn")


def generate_triplet(x, class1, class2):
    cong = class1[np.random.randint(0, class1.shape[0])]
    incong1 = class2[np.random.randint(0, class2.shape[0])]
    return np.array((x, cong, incong1))


def prepare_datasets(batch_size=30):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # DOWNLOAD DATA

    # fetch dataset
    wine = fetch_ucirepo(id=109)

    # data (as pandas dataframes)
    X = wine.data.features.to_numpy()  # (178, 13)
    y = wine.data.targets.to_numpy()  # 1, 2 or 3

    dataset_arr = np.hstack((X, y))

    # PREPARE DATA

    # divide into classes
    class1 = dataset_arr[:59]
    class2 = dataset_arr[59:130]
    class3 = dataset_arr[130:]

    # nahodne preusporiadanie so seedom
    np.random.seed(4742)
    np.random.shuffle(class1)
    np.random.seed(4742)
    np.random.shuffle(class2)
    np.random.seed(4742)
    np.random.shuffle(class3)

    # test dataset
    class_test_size = 15
    test_dataset = np.vstack(
        (class1[:class_test_size], class2[:class_test_size], class3[:class_test_size]))
    print(f"test data size: {test_dataset.shape}")
    test_dataloader = DataLoader(torch.tensor(test_dataset).to(
        device), batch_size=batch_size, shuffle=True)

    # train dataset
    use_of_x_times = 25
    dataset_size = 6650
    dataset_triplets = np.empty((dataset_size, 3, 14))
    class1, class2, class3 = class1[class_test_size:], class2[class_test_size:], class3[class_test_size:]
    position = 0

    for i in range(use_of_x_times):

        for x in class1:
            dataset_triplets[position] = generate_triplet(x, class1, class2)
            position += 1
            dataset_triplets[position] = generate_triplet(x, class1, class3)
            position += 1

        for x in class2:
            dataset_triplets[position] = generate_triplet(x, class2, class1)
            position += 1
            dataset_triplets[position] = generate_triplet(x, class2, class3)
            position += 1

        for x in class3:
            dataset_triplets[position] = generate_triplet(x, class3, class1)
            position += 1
            dataset_triplets[position] = generate_triplet(x, class3, class2)
            position += 1

    print(f"train data size: {dataset_triplets.shape}")
    train_dataloader = DataLoader(torch.tensor(dataset_triplets).to(
        device), batch_size=batch_size, shuffle=True)

    # som dataset
    som_dataloader = DataLoader(torch.tensor(np.concatenate(
        (class1, class2, class3))).to(device), shuffle=True)
    print(f"som data size: {len(som_dataloader)}")

    return train_dataloader, test_dataloader, som_dataloader, device
    
