from default_config import basic_cfg
import audiomentations as AA

cfg = basic_cfg
cfg.train_df = cfg.data_dir + "train_meta_4folded_v1.csv"
cfg.val_df = cfg.data_dir + "train_soundscape_labels_v2.csv"

# dataset
cfg.min_rating = 0.0

cfg.wav_crop_len = 30  # seconds

cfg.lr = 0.001
cfg.epochs = 20
cfg.batch_size = 16
cfg.batch_size_val = 1
cfg.dataset = "ps_ds_7"
cfg.model = "ps_model_11"
cfg.backbone = "tf_efficientnetv2_s_in21k"

cfg.label_smoothing = 0.01

cfg.num_workers = 32

cfg.save_val_data = True
cfg.mixed_precision = True

cfg.mixup = 1
cfg.mixup2 = 0
cfg.mix_beta = 1

cfg.mel_norm = True

cfg.window_size = 1024
cfg.hop_size = 320
cfg.fmin = 50
cfg.fmax = 14000
cfg.mel_bins = 64
cfg.top_db = None

cfg.train_aug = AA.Compose(
    [
        AA.AddShortNoises(
            sounds_path="/input/ff1010bird_nocall/nocall",
            min_snr_in_db=0,
            max_snr_in_db=3,
            p=0.5,
            lru_cache_size=10,
            min_time_between_sounds=4.0,
            max_time_between_sounds=16.0,
        ),
        AA.AddBackgroundNoise(
            sounds_path="/input/train_soundscapes/nocall",
            min_snr_in_db=0,
            max_snr_in_db=3,
            p=0.25,
            lru_cache_size=3,
        ),
        AA.AddBackgroundNoise(
            sounds_path="/input/aicrowd2020_noise_30sec/noise_30sec",
            min_snr_in_db=0,
            max_snr_in_db=3,
            p=0.25,
            lru_cache_size=450,
        ),
    ]
)
