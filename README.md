# SimGANs: Simulator-Based Generative Adversarial Networks for ECG Synthesis to Improve Deep ECG Classification

Pytorch implementation of [SimGANs: Simulator-Based Generative Adversarial Networks for ECG Synthesis to Improve Deep ECG Classification](http://arxiv.org/)

## Usage

To train a SimDCGAN on the MIT-BIH training data:

    $ python3 sim_gan/gan_models/train_sim_gan.py --GAN_TYPE <gan_type> --MODEL_DIR <model_dir> --BEAT_TYPE <beat_type> --BATCH_SIZE <batch_size> --NUM_ITERATIONS <num_iterations>

Where gan_type is one of the strings: {SimDCGAN, SimVGAN}

To train a  Regular VanillaGAN or DCGAN on the MIT-BIH training data:

    $ python3 sim_gan/gan_models/train_gan.py --GAN_TYPE <gan_type> --MODEL_DIR <model_dir> --BEAT_TYPE <beat_type> --BATCH_SIZE <batch_size> --NUM_ITERATIONS <num_iterations>

Where gan_type is one of the strings: {DCGAN, VGAN}

## Authors

Tomer Golany, Daniel Freedman and Kira Radinsky