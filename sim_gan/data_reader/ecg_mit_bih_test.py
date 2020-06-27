import unittest
from sim_gan.data_reader import ecg_mit_bih
import logging


class TestECGMitBihDataset(unittest.TestCase):
    def test_keys(self):
        ecg_mit_bih_ds = ecg_mit_bih.ECGMitBihDataset()
        train = ecg_mit_bih_ds.train_heartbeats
        test = ecg_mit_bih_ds.test_heartbeats

        for hb in train:
            self.assertIn('aami_label_str', hb)
        for hb in test:
            self.assertIn('aami_label_str', hb)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
