from torch.utils.data import Dataset
import numpy as np
import torch
import os
from sim_gan.data_reader import pickle_data
from sim_gan.dynamical_model import typical_beat_params, equations
from sim_gan.data_reader import ecg_mit_bih
from sim_gan.data_reader import dataset_configs, heartbeat_types
import logging
from enum import Enum
from sim_gan.gan_models.models import dcgan
from sim_gan.gan_models.models import vanila_gan
from sim_gan.gan_models.models import sim_gan_euler
from sim_gan.gan_models.models import wgan
from sim_gan.gan_models.models import refined_gan


class GanType(Enum):
    DCGAN = 0
    ODE_GAN = 1
    SIMULATOR = 2
    VANILA_GAN = 3
    VANILA_GAN_ODE = 4
    NOISE = 5
    WGAN = 6
    ODE_WGAN = 7
    REFINED_GAN = 8


class EcgHearBeatsDatasetPytorch(Dataset):
    """ECG heart beats dataset for Pytorch usage."""

    def __init__(self, configs, transform=None, pickle_data_dir=None):
        """Creates a new EcgHearBeatsDatasetPytorch object.
        :param configs: dataset_configs.DatasetConfigs object which determines the dataset configurations.
        :param transform: pytorch different transformations.
        """

        if not isinstance(configs, dataset_configs.DatasetConfigs):
            raise ValueError("configs input is not of type DatasetConfigs. instead: {}".format(type(configs)))

        self.configs = configs
        if pickle_data_dir is not None and os.path.exists(pickle_data_dir + '/ecg_mit_bih.pickle'):
            mit_bih_dataset = pickle_data.load_ecg_mit_bih_from_pickle(pickle_data_dir)
        else:
            mit_bih_dataset = ecg_mit_bih.ECGMitBihDataset()

        self.partition = configs.partition
        if configs.partition == dataset_configs.PartitionNames.train.name:
            self.data = mit_bih_dataset.train_heartbeats
        else:
            assert configs.partition == dataset_configs.PartitionNames.test.name
            self.data = mit_bih_dataset.test_heartbeats

        if configs.only_take_heartbeat_of_type is not None:
            if configs.only_take_heartbeat_of_type in heartbeat_types.AAMIHeartBeatTypes.__members__:
                assert configs.only_take_heartbeat_of_type == configs.classified_heartbeat
                self.data = np.array([sample for sample in self.data if sample['aami_label_str'] ==
                                      configs.only_take_heartbeat_of_type])
            else:
                assert configs.only_take_heartbeat_of_type == heartbeat_types.OTHER_HEART_BEATS
                self.data = np.array([sample for sample in self.data if sample['aami_label_str'] !=
                                      configs.classified_heartbeat])

        if configs.add_data_from_gan:
            num_examples_to_add = configs.gan_configs.num_examples_to_add
            generator_checkpoint_path = configs.gan_configs.checkpoint_path
            generator_beat_type = configs.gan_configs.beat_type
            assert generator_beat_type == configs.classified_heartbeat
            gan_type = configs.gan_configs.gan_type

            self.add_data_from_gan(gan_type, generator_beat_type, generator_checkpoint_path, num_examples_to_add)

        # consts:
        self.transform = transform

    def add_data_from_gan(self, gan_type, generator_beat_type, generator_checkpoint_path, num_examples_to_add):
        """

        :param gan_type:
        :param generator_beat_type:
        :param generator_checkpoint_path:
        :param num_examples_to_add:
        :return:
        """
        logging.info("Adding {} samples of type {} from GAN {}".format(num_examples_to_add, generator_beat_type,
                                                                       gan_type))
        logging.info("Size of {} data before additional data from GAN: {}".format(self.partition, len(self)))
        logging.info(
            "#N: {}\t #S: {}\t #V: {}\t #F: {}\t".format(self.len_beat('N'), self.len_beat('S'),
                                                         self.len_beat('V'), self.len_beat('F')))
        if num_examples_to_add > 0:
            if gan_type == GanType.DCGAN:
                gNet = dcgan.DCGenerator(0)
                self.add_beats_from_generator(gNet, num_examples_to_add,
                                              generator_checkpoint_path,
                                              generator_beat_type)
            elif gan_type == GanType.ODE_GAN:
                gNet = sim_gan_euler.DCGenerator(0)
                self.add_beats_from_generator(gNet, num_examples_to_add,
                                              generator_checkpoint_path,
                                              generator_beat_type)
            elif gan_type == GanType.SIMULATOR:
                self.add_beats_from_simulator(num_examples_to_add, generator_beat_type)

            elif gan_type == GanType.VANILA_GAN:
                gNet = vanila_gan.VGenerator(0)
                self.add_beats_from_generator(gNet, num_examples_to_add,
                                              generator_checkpoint_path,
                                              generator_beat_type)

            elif gan_type == GanType.VANILA_GAN_ODE:
                gNet = vanila_gan.VGenerator(0)
                self.add_beats_from_generator(gNet, num_examples_to_add,
                                              generator_checkpoint_path,
                                              generator_beat_type)
            elif gan_type == GanType.NOISE:
                self.add_noise(num_examples_to_add, generator_beat_type)

            elif gan_type == GanType.WGAN:
                gNet = wgan.WGenerator(0)
                self.add_beats_from_generator(gNet, num_examples_to_add,
                                              generator_checkpoint_path,
                                              generator_beat_type)
            elif gan_type == GanType.ODE_WGAN:
                gNet = wgan.WGenerator(0)
                self.add_beats_from_generator(gNet, num_examples_to_add,
                                              generator_checkpoint_path,
                                              generator_beat_type)
            elif gan_type == GanType.REFINED_GAN:
                gNet = refined_gan.RefineGenerator(0)
                self.add_beats_from_generator(gNet, num_examples_to_add,
                                              generator_checkpoint_path,
                                              generator_beat_type, input_is_noise=False)

            else:
                raise ValueError("Invalid gan type")

    def __len__(self):
        return len(self.data)

    def len_beat(self, beat_type):
        if beat_type not in heartbeat_types.AAMIHeartBeatTypes.__members__:
            raise ValueError("Invalid heart-beat type: {}".format(beat_type))
        return len(np.array([sample for sample in self.data if sample['aami_label_str'] == beat_type]))

    def __getitem__(self, idx):
        """Overload [] operator, to get the sample at index idx.

        :param idx:
        :return:
        """

        sample = self.data[idx]

        if self.configs.lstm_setting:
            # TODO(tomergolany): add support for second lead in LSTM case.
            heartbeat_main_lead = np.array([sample['cardiac_cycle'][i:i + 5] for i in range(0, 215, 5)])
        else:
            heartbeat_main_lead = sample['cardiac_cycle']

        # heartbeat_second_lead = sample['cardiac_cycle_other_lead']
        # second_lead_name = sample['other_lead_name']
        heartbeat_label_str = sample['aami_label_str']
        if not self.configs.one_vs_all:
            sample = {'cardiac_cycle': heartbeat_main_lead, 'beat_type': heartbeat_label_str,
                      'label': np.array(sample['aami_label_one_hot']),
                      # 'cardiac_cycle_other_lead': heartbeat_second_lead,
                      #'other_lead_name': second_lead_name
            }
        else:
            if heartbeat_label_str == self.configs.classified_heartbeat:
                sample = {'cardiac_cycle': heartbeat_main_lead, 'beat_type': heartbeat_label_str,
                          'label': np.array([1, 0]),
                          # 'cardiac_cycle_other_lead': heartbeat_second_lead,
                          # 'other_lead_name': second_lead_name
                          }
            else:
                sample = {'cardiac_cycle': heartbeat_main_lead, 'beat_type': heartbeat_label_str,
                          'label': np.array([0, 1]),
                          # 'cardiac_cycle_other_lead': heartbeat_second_lead,
                          # 'other_lead_name': second_lead_name
                          }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def add_beats_from_generator(self, generator_model, num_beats_to_add, checkpoint_path, beat_type, input_is_noise=True):
        logging.info("Adding data from generator: {}. number of beats to add: {}\t"
                     "checkpoint path: {}\t beat type: {}".format(generator_model, num_beats_to_add, checkpoint_path,
                                                                  beat_type))
        checkpoint = torch.load(checkpoint_path)
        generator_model.load_state_dict(checkpoint['generator_state_dict'])
        generator_model.eval()
        # discriminator_model.load_state_dict(checkpoint['discriminator_state_dict'])
        with torch.no_grad():
            # input_noise = torch.Tensor(np.random.normal(0, 1, (num_beats_to_add, 100)))
            if input_is_noise:
                logging.info("Input to GENERATOR is NOISE.")
                input_noise = torch.Tensor(np.random.uniform(0, 1, (num_beats_to_add, 100)))
            else:
                logging.info("Input to GENERATOR is data from the simulator.")
                input_noise = self.add_beats_from_simulator(num_beats_to_add, beat_type)
                input_noise = np.array([x['cardiac_cycle'] for x in input_noise])
                input_noise = torch.from_numpy(input_noise).float()

            output_g = generator_model(input_noise)
            output_g = output_g.numpy()
            output_g = np.array(
                [{'cardiac_cycle': x, 'aami_label_str': beat_type, 'aami_label_one_hot':
                    np.array([1, 0]), 'cardiac_cycle_other_lead': None, 'other_lead_name': None} for x
                 in output_g])
            # plt.plot(output_g[0]['cardiac_cycle'])
            # plt.show()
            self.additional_data_from_gan = output_g
            self.data = np.concatenate((self.data, output_g))
            print("Length of {} data after adding from generator is {}".format(self.partition, len(self.data)))

    def add_beats_from_simulator(self, num_beats_to_add, beat_type):
        beat_params = typical_beat_params.beat_type_to_typical_param[beat_type]
        noise_param = (np.random.normal(0, 0.1, (num_beats_to_add, 15)))
        params = 0.01 * noise_param + beat_params
        sim_beats = equations.generate_batch_of_beats_numpy(params)
        sim_beats = np.array(
            [{'cardiac_cycle': x, 'aami_label_str': beat_type, 'aami_label_one_hot': np.array([1, 0])} for x
             in sim_beats])
        self.additional_data_from_simulator = sim_beats
        self.data = np.concatenate((self.data, sim_beats))
        print("Length of train samples after adding from simulator is {}".format(len(self.data)))
        return sim_beats

    def add_noise(self, n, beat_type):
        input_noise = np.random.normal(0, 1, (n, 216))

        input_noise = np.array(
            [{'cardiac_cycle': x, 'beat_type': beat_type, 'label': self.beat_type_to_one_hot_label[beat_type]} for x
             in input_noise])
        self.train = np.concatenate((self.train, input_noise))


class Scale(object):
    def __call__(self, sample):
        heartbeat, label = sample['cardiac_cycle'], sample['label']
        heartbeat = scale_signal(heartbeat)
        return {'cardiac_cycle': heartbeat,
                'label': label,
                'beat_type': sample['beat_type'],
                # 'cardiac_cycle_other_lead': sample['cardiac_cycle_other_lead'],
                # 'other_lead_name': sample['other_lead_name']
                }


def scale_signal(signal, min_val=-0.01563, max_val=0.042557):
    """

    :param min:
    :param max:
    :return:
    """
    scaled = np.interp(signal, (signal.min(), signal.max()), (min_val, max_val))
    return scaled


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        heartbeat, label = sample['cardiac_cycle'], sample['label']
        return {'cardiac_cycle': (torch.from_numpy(heartbeat)).double(),
                'label': torch.from_numpy(label),
                'beat_type': sample['beat_type'],
                }


if __name__ == "__main__":
    ds = EcgHearBeatsDatasetPytorch()
