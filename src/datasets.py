import os
import numpy as np 
import torch
import pandas as pd
import torch.utils.data as data 
import surfa as sf
import shutil

class CondWarpDataset(data.Dataset):
    """
    Class for loading dataset for conditional warp modelling.
    """

    def __init__(self, 
                 df: pd.DataFrame, 
                 dataset_root: str,
                 volume_args: dict,
                 min_max_args: dict,
    ) -> None:

        self.df = df
        self.dataset_root = dataset_root
        self.volume_args = volume_args

        self.min_age = min_max_args["age_at_scan"][0]
        self.max_age = min_max_args["age_at_scan"][1]

        self.min_age_floor = min_max_args["min_age_floor"]
        self.mean_tracker_width = min_max_args["mean_tracker_width"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        image_path = os.path.join(self.dataset_root, self.df.iloc[idx]["scan_path"])
        age_at_scan = self.df.iloc[idx]["age_at_scan"]
        norm_age_at_scan = (age_at_scan - self.min_age) / (self.max_age - self.min_age)
        age_bucket = self.df.iloc[idx]["age_bucket"]

        image = sf.load_volume(image_path)

        if self.volume_args["random_gamma"]:
            gamma_value = np.exp(
                np.random.uniform(
                    self.volume_args["random_gamma_params"][0],
                    self.volume_args["random_gamma_params"][1],
                )
            )
        
            image = image ** gamma_value
    
        image = image / image.percentile(self.volume_args["percentile_normalisation"])

        if self.volume_args["random_noise"]:
            noise_mean = self.volume_args["random_noise_params"][0]
            noise_std = self.volume_args["random_noise_params"][1]
            noise = np.random.normal(noise_mean, noise_std, size = image.shape)
            image = image + noise      

        image = image.clip(0, 1.1)
        image = image.conform(
            voxsize = self.volume_args["vox_size"],
            dtype = self.volume_args["dtype"],
            method = self.volume_args["method"],
            orientation = self.volume_args["orientation"],
        )

        image = image.reshape(self.volume_args["target_shape"])
        geometry = np.array(image.geom.vox2world)
        image = np.array(image.data)
        image = torch.from_numpy(image.copy()).unsqueeze(0)

        means_idx_0 = int((age_at_scan - self.min_age_floor) // self.mean_tracker_width)
        means_idx_1 = means_idx_0 + 1

        prop_means_idx_1 = ((age_at_scan - self.min_age_floor) % self.mean_tracker_width) / self.mean_tracker_width
        prop_means_idx_0 = 1 - prop_means_idx_1

        sample = {
            "image": image,
            "scan_path" : image_path,
            "age_at_scan" : age_at_scan,
            "norm_age_at_scan" : norm_age_at_scan,
            "param" : norm_age_at_scan,
            "param_orig" : age_at_scan,
            "age_bucket" : age_bucket,
            "means_idx_0" : means_idx_0,
            "prop_means_idx_0" : prop_means_idx_0,
            "means_idx_1" : means_idx_1,
            "prop_means_idx_1" : prop_means_idx_1,
            "geometry" : geometry,
            "orientation" : self.volume_args["orientation"],
            "vox_size" : self.volume_args["vox_size"],
        }
        return sample

def get_dls_and_template(
    dataset: data.Dataset,
    dataset_root: str,
    template_path: str,
    batch_size: int,
    num_workers: int = 8,
    dataset_file_path: str = None,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    num_loaders: int = 2,
    loader_proportions: list = None,
    max_samples_per_subject: int = None,
    downsample: bool = False,
    downsample_dim: int = 128,
    dtype: str = "float32",
    age_buckets: int = 10,
    mean_tracker_width: int = 1,
    runtime_dir: str = None,
    run_id: int = None,
    predefined_split: bool = False,
    loader_file_paths: list = None,
    eval_mode: bool = False,
    template_avg_int_shift_factor: float = 1.0,
    percentile_normalisation: float = 99.9,
    random_gamma: bool = False,
    random_gamma_params: dict = None,
    random_noise: bool = False,
    random_noise_params: dict = None,
) -> list:
    dls_and_template = []

    assert dataset_file_path is not None, \
        "dataset_file_path must be specified"
    
    dataset_csv_type = dataset_file_path.split(".")[-1]
    
    if dataset_csv_type == "csv":
        df = pd.read_csv(dataset_file_path)
    elif dataset_csv_type in ["xlsx", "xls"]:
        df = pd.read_excel(dataset_file_path)
    elif dataset_csv_type == "json":
        df = pd.read_json(dataset_file_path)
    elif dataset_csv_type == "feather":
        pd.read_feather(dataset_file_path)
    elif dataset_csv_type == "pkl":
        df = pd.read_pickle(dataset_file_path)
    elif dataset_csv_type == "tsv":
        df = pd.read_csv(dataset_file_path, sep = "\t")
    else:
        raise ValueError("Dataset file type not supported")

    if not predefined_split:
        if loader_proportions is None:
            loader_proportions = [1] * num_loaders
        loader_proportions = [x / sum(loader_proportions) for x in loader_proportions]

        df_grouped = df.groupby("participant_id")["participant_id"].head(1)
        df_grouped = df_grouped.sample(frac = 1.).reset_index(drop = True)
        df_grouped_len = len(df_grouped)
        df_grouped_len_list = [int(x * df_grouped_len) for x in loader_proportions]
        df_grouped_len_list[-1] = df_grouped_len - sum(df_grouped_len_list[:-1])
        df_grouped_list = np.split(df_grouped, np.cumsum(df_grouped_len_list))[:-1]

        if max_samples_per_subject is not None:
            df = df.groupby("participant_id").head(max_samples_per_subject)
    
    min_age = df["age_at_scan"].min()
    max_age = df["age_at_scan"].max()

    print(f"Min age: {min_age}")
    print(f"Max age: {max_age}")

    df["age_bucket"] = pd.cut(
        df["age_at_scan"], 
        bins = age_buckets, 
        labels = [i for i in range(age_buckets)]
    )

    min_edu_level = df["education_level"].min()
    max_edu_level = df["education_level"].max()

    min_age_floor = int(np.floor(min_age))
    max_age_ceil = int(np.ceil(max_age))
    age_range = max_age_ceil - min_age_floor
    
    assert mean_tracker_width in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], \
        "means_width must be an integer between 1 and 10."
    
    if age_range % mean_tracker_width == 0:
        nb_means = int(age_range // mean_tracker_width + 1)
    else:
        nb_means = int(age_range // mean_tracker_width + 2)
    
    min_max_args = {
        "age_at_scan" : (min_age, max_age),
        "education_level" : (min_edu_level, max_edu_level),
        "min_age_floor" : min_age_floor,
        "max_age_ceil" : max_age_ceil,
        "age_range" : age_range,
        "nb_means" : nb_means,
        "mean_tracker_width" : mean_tracker_width,
    }

    if not predefined_split:
        df_list = []
        for idx, df_grouped in enumerate(df_grouped_list):
            _loader_df = df[df["participant_id"].isin(df_grouped.values)]
            _loader_df = _loader_df.sample(frac = 1.).reset_index(drop = True)
            df_list.append(_loader_df)
    else:
        assert len(loader_file_paths) == num_loaders, \
            "Number of loader file paths must match number of loaders"
        
        assert all([x.split(".")[-1] == "csv" for x in loader_file_paths]), \
            "Loader file paths must be csv files"
        
        df_list = []
        
        for loader_file_path in loader_file_paths:
            _loader_df = pd.read_csv(loader_file_path)
            df_list.append(_loader_df)
        
        assert all([len(x.columns) == len(df_list[0].columns) for x in df_list]), \
            "All loader csv files must have the same number of columns"
        
        assert all([len(x) > 0 for x in df_list]), \
            "All loader csv files must have at least one row"
        
        for idx in range(1, len(df_list)):
            assert len(df_list[0]) > len(df_list[idx]), \
                "The first loader csv file must have more rows than the other loader csv files"
        
        _df_len = 0
        for idx in range(len(df_list)):
            _df_len += len(df_list[idx])
        
        df_list = [x.sample(frac = 1.).reset_index(drop = True) for x in df_list]

    template = sf.load_volume(template_path)
    template = template / template.percentile(percentile_normalisation)
    template = template * template_avg_int_shift_factor
    template = template.clip(0, 1.1)
    target_shape = np.clip(np.ceil(np.array(template.shape[:3]) / 64).astype(int) * 64, 192, 320)
    vox_size = 1.0
    
    if downsample:
        
        max_elem = np.max(target_shape)
        
        assert downsample_dim % 64 == 0, \
            "Downsample dimension must be a multiple of 64"
        
        assert downsample_dim <= max_elem, \
            "Downsample dimension must be less than or equal to max element of the volume conformed to 1mm"
        
        vox_size = vox_size * max_elem / downsample_dim
        target_shape = np.array([downsample_dim for _ in range(len(template.shape))]).astype(int)
    
    template = template.conform(
        voxsize = vox_size, 
        dtype = dtype, 
        method = "linear",
        orientation = "RAS",
    )

    print(f"Voxel size: {vox_size} mm")
    print(f"Image shape: {target_shape}")
    print(f"Image geometry:\n{np.array(template.geom.vox2world)}")

    template = template.reshape(target_shape)
    volume_args = {
        "vox_size" : vox_size,
        "target_shape" : target_shape,
        "method" : "linear",
        "orientation" : "RAS",
        "dtype" : dtype,
        "geometry" : np.array(template.geom.vox2world),
        "percentile_normalisation" : percentile_normalisation,
        "random_gamma" : random_gamma,
        "random_gamma_params" : random_gamma_params,
        "random_noise" : random_noise,
        "random_noise_params" : random_noise_params,
    }
    template = np.array(template.data)

    if not predefined_split:
        df_save_dir = f"{runtime_dir}/../logs/{run_id}"
        for idx, _df in enumerate(df_list):
            _filename = f"{df_save_dir}/run_{run_id}_loader_{idx}.csv"
            _df.to_csv(_filename, index = False)
    else:
        if not eval_mode:
            df_save_dir = f"{runtime_dir}/../logs/{run_id}"
            for idx, loader_file_path in enumerate(loader_file_paths):
                _filename = f"{df_save_dir}/run_{run_id}_loader_{idx}.csv"
                shutil.copyfile(loader_file_path, _filename)

    for idx in range(num_loaders):
        
        if idx != 0:
            volume_args["random_gamma"] = False
            volume_args["random_noise"] = False
        
        if volume_args["random_gamma"]:
            print(f"Random gamma enabled for loader {idx} with parameters for beta drawn from uniform (gamma = exp(beta)): {volume_args['random_gamma_params']}")
        if volume_args["random_noise"]:
            print(f"Random noise enabled for loader {idx} with parameters mean and std: {volume_args['random_noise_params']}")
        
        df = df_list[idx]
        _dataset = dataset(
            df, 
            dataset_root, 
            volume_args, 
            min_max_args
        )
        
        dls_and_template.append(
            data.DataLoader(
                _dataset,
                batch_size = batch_size,
                num_workers = num_workers,
                shuffle = shuffle,
                drop_last = drop_last,
                pin_memory = pin_memory,
            )
        )
    
    dls_and_template.append((torch.from_numpy(template.copy()), None))
    dls_and_template.append(min_max_args)
    dls_and_template.append(volume_args)
    
    return dls_and_template

    

