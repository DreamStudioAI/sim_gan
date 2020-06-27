import pickle
from sim_gan.data_reader import ecg_mit_bih
import logging


def load_ecg_input_from_pickle(pickle_dir):
    with open(pickle_dir + '/train_beats.pickle', 'rb') as handle:
        train_beats = pickle.load(handle)
    with open(pickle_dir + '/val_beats.pickle', 'rb') as handle:
        validation_beats = pickle.load(handle)
    with open(pickle_dir + '/test_beats.pickle', 'rb') as handle:
        test_beats = pickle.load(handle)
    return train_beats, validation_beats, test_beats


def save_ecg_mit_bih_to_pickle(pickle_dir):
    print("start pickling:")
    with open(pickle_dir + '/ecg_mit_bih.pickle', 'wb') as output:
        ecg_ds = ecg_mit_bih.ECGMitBihDataset()
        pickle.dump(ecg_ds, output, pickle.HIGHEST_PROTOCOL)
        logging.info("Done pickling")


def load_ecg_mit_bih_from_pickle(pickle_dir):
    logging.info("Loading ecg-mit-bih from pickle...")
    with open(pickle_dir + '/ecg_mit_bih.pickle', 'rb') as handle:
        ecg_ds = pickle.load(handle)
        logging.info("Loaded successfully...")
        return ecg_ds
