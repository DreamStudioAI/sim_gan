"""Function that support building dataset for ECG heartbeats."""
import torchvision.transforms as transforms
from sim_gan.data_reader.ecg_dataset_pytorch import ToTensor
from sim_gan.data_reader import ecg_dataset_pytorch
import torch
import logging


def build(train_config, dataset_train_configs, dataset_test_configs):
    """Build PyTorch train and test data-loaders.

    :param train_config: Train configurations
    :param dataset_train_configs
    :param dataset_test_configs
    :return:
    """
    add_from_gan = dataset_train_configs.add_data_from_gan
    batch_size = train_config.batch_size
    composed = transforms.Compose([ToTensor()])

    train_dataset = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs=dataset_train_configs, transform=composed)
    test_dataset = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs=dataset_test_configs, transform=composed)

    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=300,
                                                 shuffle=True, num_workers=1)

    #
    # Check if to add data from GAN:
    #
    if add_from_gan:
        logging.info("Size of training data after additional data from GAN: {}".format(len(train_dataset)))
        logging.info("#N: {}\t #S: {}\t #V: {}\t #F: {}\t".format(train_dataset.len_beat('N'), train_dataset.len_beat('S'),
                                                                train_dataset.len_beat('V'), train_dataset.len_beat('F')))
    else:
        logging.info("No data is added. Train set size: ")
        logging.info("#N: {}\t #S: {}\t #V: {}\t #F: {}\t".format(train_dataset.len_beat('N'), train_dataset.len_beat('S'),
                                                                  train_dataset.len_beat('V'), train_dataset.len_beat('F')))
        logging.info("test set size: ")
        logging.info("#N: {}\t #S: {}\t #V: {}\t #F: {}\t".format(test_dataset.len_beat('N'), test_dataset.len_beat('S'),
                                                                  test_dataset.len_beat('V'), test_dataset.len_beat('F')))

    if train_config.weighted_sampling:
        weights_for_balance = train_dataset.make_weights_for_balanced_classes()
        weights_for_balance = torch.DoubleTensor(weights_for_balance)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=weights_for_balance,
            num_samples=len(weights_for_balance),
            replacement=True)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                 num_workers=1, sampler=sampler)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                 num_workers=1, shuffle=True)

    return train_data_loader, testdataloader