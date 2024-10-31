import os
import numpy as np 
import torch
import pandas as pd
import torch.utils.data as data 
import surfa as sf
# from typing import Tuple, List, Dict, Any, Union, Optional, Iterable
import shutil

# class SynthDataset(data.Dataset):
#     """
#     Dataset class for loading synthetic data."""

#     def __init__(self, 
#                  paths: list, 
#                  root: str, 
#                  dtype: str = "float32",
#                  param_range: tuple = (70, 89),
#                  downsample = False,
#                  downsample_dim: int = 192,
#                  ) -> None:
#         """
#         Args:
#             paths (list): list of filenames.
#             root (string): root directory of dataset.
#         """
#         self.paths = paths
#         self.root = root

#         self.max_param = param_range[1]
#         self.min_param = param_range[0]
#         self.param_diff = self.max_param - self.min_param

#         self.dtype = dtype

#         if downsample:
#             self.downsample = True
#             self.downsample_dim = downsample_dim
#         else:
#             self.downsample = False

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = self.paths[idx]
#         image = np.load(os.path.join(self.root, img_name)) / 255

#         if self.downsample:
#             image = skimage.transform.resize(
#                 image, 
#                 (self.downsample_dim, self.downsample_dim),
#                 order = 1,
#                 preserve_range = True,
#                 mode = "edge",
#             )
            
#             # image = cv2.resize(
#             #     image, 
#             #     (self.downsample_dim, self.downsample_dim),
#             #     interpolation = cv2.INTER_LINEAR,
#             # )
#         image = torch.from_numpy(image.astype(self.dtype)).unsqueeze(0)
#         try:
#             param_orig = int(img_name.split('_')[-2])
#             param = float(param_orig - self.min_param) / self.param_diff
#         except:
#             raise ValueError("No age in filename")
#         sample = {"image": image,
#                   "param": param,
#                   "name": img_name,
#                   "param_orig": param_orig}

#         return sample

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

        # self.marital_status_dict = {
        #     "Married": 0, 
        #     "Widowed": 1, 
        #     "Divorced": 2, 
        #     "Never married": 3,
        #     "Unknown": 4,
        # }
        # self.ethnicity_dict = {
        #     "Not Hisp/Latino": 0,
        #     "Hisp/Latino": 1,
        # }
        # self.race_dict = {
        #     "White": 0,
        #     "Black": 1,
        #     "Asian": 2,
        #     "More than one": 3,
        #     "Unknown": 4,
        #     "Am Indian/Alaskan": 5,
        # }
        # self.diag_dict = {
        #     "CN": 0,
        #     "MCI": 1,
        #     "AD": 2,
        # }
        # self.sex_dict = {
        #     "M": 0,
        #     "F": 1,
        # }

        self.min_age = min_max_args["age_at_scan"][0]
        self.max_age = min_max_args["age_at_scan"][1]

        # self.min_edu_level = min_max_args["education_level"][0]
        # self.max_edu_level = min_max_args["education_level"][1]

        self.min_age_floor = min_max_args["min_age_floor"]
        self.mean_tracker_width = min_max_args["mean_tracker_width"]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        image_path = os.path.join(self.dataset_root, self.df.iloc[idx]["scan_path"])
        # scan_id = self.df.iloc[idx]["scan_id"]
        # participant_id = self.df.iloc[idx]["participant_id"]
        # alternate_id = self.df.iloc[idx]["alternate_id"]
        # sex = self.sex_dict[self.df.iloc[idx]["sex"]]
        # age_initial = self.df.iloc[idx]["age_initial"]
        age_at_scan = self.df.iloc[idx]["age_at_scan"]
        norm_age_at_scan = (age_at_scan - self.min_age) / (self.max_age - self.min_age)
        # edu_level = self.df.iloc[idx]["education_level"]
        # norm_edu_level = (edu_level - self.min_edu_level) / (self.max_edu_level - self.min_edu_level)
        # marital_status = self.marital_status_dict[self.df.iloc[idx]["marital_status"]]
        # ethnic_category = self.ethnicity_dict[self.df.iloc[idx]["ethnic_category"]]
        # racial_category = self.race_dict[self.df.iloc[idx]["racial_category"]]
        # diagnosis = self.diag_dict[self.df.iloc[idx]["diagnosis"]]
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
            # "scan_id" : scan_id,
            # "participant_id" : participant_id,
            # "alternate_id" : alternate_id,
            # "sex" : sex,
            # "age_initial" : age_initial,
            "age_at_scan" : age_at_scan,
            "norm_age_at_scan" : norm_age_at_scan,
            # "edu_level" : edu_level,
            # "norm_edu_level" : norm_edu_level,
            # "marital_status" : marital_status,
            # "ethnic_category" : ethnic_category,
            # "racial_category" : racial_category,
            # "diagnosis" : diagnosis, 
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

# class InverseWarpDataset(data.Dataset):
#     """
#     Class for loading dataset for inverse warp modelling.
#     """

#     def __init__(self, 
#                  df: pd.DataFrame, 
#                  eval_output_dir: str,
#                  dtype: str = "float32",
#                  use_nifti: bool = False,
#     ) -> None:

#         self.df = df
#         self.eval_output_dir = eval_output_dir
#         self.dtype = getattr(torch, dtype)
#         self.use_nifti = use_nifti

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
        
#         scan_id = self.df.iloc[idx]["scan_id"]
#         participant_id = self.df.iloc[idx]["participant_id"]
#         alternate_id = self.df.iloc[idx]["alternate_id"]
#         age_at_scan = self.df.iloc[idx]["age_at_scan"]
#         eval_filename = self.df.iloc[idx]["eval_filename"]
#         eval_filename_path = os.path.join(self.eval_output_dir, eval_filename)


#         if self.use_nifti:
#             cond_temp_warp = nib.load(os.path.join(self.eval_output_dir, "cond_temp_warp", eval_filename))
#             flow_st_1 = nib.load(os.path.join(self.eval_output_dir, "flow_st_1", eval_filename)).get_fdata()
#             geometry = cond_temp_warp.affine
#             cond_temp_warp = cond_temp_warp.get_fdata()
#             cond_temp_warp = cond_temp_warp.transpose(3, 0, 1, 2)
#             flow_st_1 = flow_st_1.transpose(3, 0, 1, 2)
#             # cond_temp_warp = sf.load_volume(os.path.join(self.eval_output_dir, "cond_temp_warp", eval_filename)).data
#             # flow_st_1 = sf.load_volume(os.path.join(self.eval_output_dir, "flow_st_1", eval_filename)).data
#             cond_temp_warp = torch.tensor(cond_temp_warp).type(self.dtype)
#             flow_st_1 = torch.tensor(flow_st_1).type(self.dtype)

#         else:

#             with open(eval_filename_path, "rb") as f:
#                 eval_output = pkl.load(f)
        
#             cond_temp_warp = torch.tensor(eval_output["cond_temp_warp"]).type(self.dtype)
#             flow_st_1 = torch.tensor(eval_output["flow_st_1"]).type(self.dtype)
#             geometry = eval_output["geometry"]
        
#         sample = {
#             "participant_id" : participant_id,
#             "alternate_id" : alternate_id,
#             "scan_id" : scan_id,
#             "age_at_scan" : age_at_scan,
#             "eval_filename" : eval_filename,
#             "cond_temp_warp" : cond_temp_warp,
#             "flow_st_1" : flow_st_1,
#             "geometry" : geometry,
#         }
#         return sample



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
    # synth_data: bool = False,
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

    # if synth_data:
    #     # synthetic data pipeline
        
    #     paths = os.listdir(dataset_root)
    #     val_size = int(0.2 * len(paths))
    #     # generate random indices for validation set
    #     val_indices = np.random.choice(len(paths), val_size, replace = False)
    #     # generate training set indices
    #     train_indices = np.array([i for i in range(len(paths)) if i not in val_indices])
    #     # generate validation set paths
    #     val_paths = [paths[i] for i in val_indices]
    #     # generate training set paths
    #     train_paths = [paths[i] for i in train_indices]
        
    #     template = np.load(template_path) / 255 
    #     template_shape = template.shape
        
    #     if downsample:
    #         downsample = not template.count(downsample_dim) == len(template.shape)
        
    #     if downsample:
    #         max_elem = np.max(template_shape)
    #         assert downsample_dim % 64 == 0, "Downsample dimension must be a multiple of 64"
    #         assert downsample_dim <= max_elem, \
    #             "Downsample dimension must be less than or equal to max element of the volume conformed to 1mm"
    #         target_shape = np.array([downsample_dim for _ in range(len(template.shape))]).astype(int)
    #         template = skimage.transform.resize(template, target_shape, order = 1, mode = "edge", preserve_range = True)
    #     template = torch.from_numpy(template.astype(dtype))
        
    #     dataloader_train = data.DataLoader(
    #         dataset(
    #             train_paths, 
    #             dataset_root,
    #             dtype = dtype,
    #             downsample = downsample,
    #             downsample_dim = downsample_dim,
    #         ), 
    #         batch_size = batch_size, 
    #         shuffle = True, 
    #         num_workers = num_workers
    #     )

    #     dls_and_template.append(dataloader_train)        
        
    #     dataloader_val = data.DataLoader(
    #         dataset(
    #             val_paths, 
    #             dataset_root,
    #             dtype = dtype,
    #             downsample = downsample,
    #             downsample_dim = downsample_dim,
    #         ), 
    #         batch_size = batch_size, 
    #         shuffle = True, 
    #         num_workers = num_workers
    #     )
        
    #     dls_and_template.append(dataloader_val) 
    #     dls_and_template.append(template)

    # else:
        # real data pipeline
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

    # if template_seg_path is not None:
    #     template_seg = sf.load_volume(template_seg_path)
    #     template_seg = template_seg.conform(
    #         voxsize = vox_size, 
    #         dtype = dtype, 
    #         method = "nearest", # should we be using nearest or linear?
    #         orientation = "RAS",
    #     )#.crop_to_bbox()

    #     template_seg = template_seg.reshape(target_shape)
    #     template_seg = np.array(template_seg.data)
    #     template_seg = template_seg.astype(dtype)

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
    
    # if template_seg_path is not None:
    #     dls_and_template.append((torch.from_numpy(template.copy()), torch.from_numpy(template_seg.copy())))
    # else:
    dls_and_template.append((torch.from_numpy(template.copy()), None))
    
    dls_and_template.append(min_max_args)
    dls_and_template.append(volume_args)
    
    return dls_and_template


# def get_inverse_warp_loaders(
#     eval_output_dir: str,
#     loader_subjects_path: str,
#     batch_size: int,
#     shuffle: bool = True,
#     drop_last: bool = True,
#     pin_memory: bool = True,
#     num_workers: int = 8,
#     num_loaders: int = 2,
#     loader_proportions: list = None,
#     predefined_split: bool = False,
#     loader_paths: list = None,
#     dtype: str = "float32",
#     runtime_dir: str = None,
#     run_id: int = None,
#     eval_mode: bool = False,
#     use_nifti: bool = False,
# ):
    
#     if predefined_split:
#         assert loader_paths is not None, \
#             "loader_paths must be specified if predefined_split is True"
#         loader_subject_df = pd.read_csv(loader_paths[0])
#     else:
#         loader_subject_df = pd.read_csv(loader_subjects_path)

#     # assert that every item in eval output dir has a corresponding item in loader subject df
#     if not use_nifti:
#         eval_output_dir_list = os.listdir(eval_output_dir)
#     else:
#         eval_output_dir_list = os.listdir(f"{eval_output_dir}/cond_temp_warp")
    
#     # concat participant_id and scan_id to get the filename
    
    
#     if not use_nifti:
#         # loader_subject_df["eval_filename"] = str(loader_subject_df["participant_id"].values) + "_" + str(loader_subject_df["scan_id"].values) + ".pkl"
#         loader_subject_df["eval_filename"] = loader_subject_df["participant_id"].astype(str) + "_" + loader_subject_df["scan_id"].astype(str) + ".pkl"
#         loader_subject_df.to_csv(f"{runtime_dir}/../logs/{run_id}/loader_subject_df_inv_warp_map.csv", index = False)
#         for filename in eval_output_dir_list:
#             assert filename in loader_subject_df["eval_filename"].values, \
#                 f"{filename} not found in loader subject df"
        
#     else:
#         # loader_subject_df["eval_filename"] = str(loader_subject_df["participant_id"].values) + "_" + str(loader_subject_df["scan_id"].values) + ".nii"
#         loader_subject_df["eval_filename"] = loader_subject_df["participant_id"].astype(str) + "_" + loader_subject_df["scan_id"].astype(str) + ".nii.gz"
#         loader_subject_df.to_csv(f"{runtime_dir}/../logs/{run_id}/loader_subject_df_inv_warp_map.csv", index = False)
#         for filename in eval_output_dir_list:
#             assert filename in loader_subject_df["eval_filename"].values, \
#                 f"{filename} not found in loader subject df"

#     if predefined_split:
#         if not eval_mode:
#             # assert loader_paths is not None, \
#             #     "loader_paths must be specified if predefined_split is True"
            
#             loader_subject_df = loader_subject_df.sample(frac = 1.).reset_index(drop = True)

#             # create a list of dataframes for each loader

#             if loader_proportions is None:
#                 num_loaders = 2
#                 loader_proportions = [0.8, 0.2]
#             else:
#                 num_loaders = len(loader_proportions)
#                 assert sum(loader_proportions) == 1, "Loader proportions must sum to 1"
            
#             df_grouped = loader_subject_df.groupby("participant_id")["participant_id"].head(1)
#             df_grouped = df_grouped.sample(frac = 1.).reset_index(drop = True)
#             df_grouped_len = len(df_grouped)
#             df_grouped_len_list = [int(x * df_grouped_len) for x in loader_proportions]
#             df_grouped_len_list[-1] = df_grouped_len - sum(df_grouped_len_list[:-1])
#             df_grouped_list = np.split(df_grouped, np.cumsum(df_grouped_len_list))[:-1]


#     if not predefined_split:
#         raise ValueError("Not implemented")
#         # df_list = []
#         # for idx, df_grouped in enumerate(df_grouped_list):
#         #     _loader_df = loader_subject_df[loader_subject_df["participant_id"].isin(df_grouped.values)]
#         #     _loader_df = _loader_df.sample(frac = 1.).reset_index(drop = True)
#         #     df_list.append(_loader_df)
#     else:
#         df_list = []
        
#         if not eval_mode:
#             for idx, df_grouped in enumerate(df_grouped_list):
#                 _loader_df = loader_subject_df[loader_subject_df["participant_id"].isin(df_grouped.values)]
#                 _loader_df = _loader_df.sample(frac = 1.).reset_index(drop = True)
#                 df_list.append(_loader_df)
        
#         else:
#             loader_loc = f"{runtime_dir}/../logs/{run_id}/inverse_warp_map/"
#             for idx in range(num_loaders):
#                 _loader_df = pd.read_csv(f"{loader_loc}run_{run_id}_loader_{idx}.csv")
#                 df_list.append(_loader_df)
        
#         # assert len(loader_paths) == num_loaders, \
#         #     "Number of loader file paths must match number of loaders"
        
#         # assert all([x.split(".")[-1] == "csv" for x in loader_paths]), \
#         #     "Loader file paths must be csv files"
        
#         # df_list = []
        
#         # for loader_path in loader_paths:
#         #     _loader_df = pd.read_csv(loader_path)
#         #     df_list.append(_loader_df)
        
#         # assert all([len(x.columns) == len(df_list[0].columns) for x in df_list]), \
#         #     "All loader csv files must have the same number of columns"
        
#         # assert all([len(x) > 0 for x in df_list]), \
#         #     "All loader csv files must have at least one row"
        
#         # for idx in range(1, len(df_list)):
#         #     assert len(df_list[0]) > len(df_list[idx]), \
#         #         "The first loader csv file must have more rows than the other loader csv files"
        
#         # _df_len = 0
#         # for idx in range(len(df_list)):
#         #     _df_len += len(df_list[idx])
        
#         # assert _df_len == len(loader_subject_df), \
#         #     "The sum of the rows of the loader csv files must equal the number of rows of the dataset csv file"
        
#         # df_list = [x.sample(frac = 1.).reset_index(drop = True) for x in df_list]
    
#     if not predefined_split:
#         raise ValueError("Not implemented")
#         # if not eval_mode:
#         #     df_save_dir = f"{runtime_dir}/../logs/{run_id}/inverse_warp_map"
#         #     for idx, _df in enumerate(df_list):
#         #         _filename = f"{df_save_dir}/run_{run_id}_loader_{idx}.csv"
#         #         _df.to_csv(_filename, index = False)
#     else:
#         if not eval_mode:
#             df_save_dir = f"{runtime_dir}/../logs/{run_id}/inverse_warp_map"
#             # create df_save_dir if it doesn't exist
#             if not os.path.exists(df_save_dir):
#                 os.makedirs(df_save_dir)
#             for idx, loader_file_path in enumerate(loader_paths):
#                 # print(loader_file_path)
#                 _filename = f"{df_save_dir}/run_{run_id}_loader_{idx}.csv"
#                 _loader_df = df_list[idx]
#                 _loader_df.to_csv(_filename, index = False)
#                 # shutil.copyfile(loader_file_path, _filename)
        
    
#     dataloaders = []
    
#     for idx in range(num_loaders):
#         df = df_list[idx]
#         _dataset = InverseWarpDataset(
#             df, 
#             eval_output_dir,
#             dtype = dtype,
#             use_nifti = use_nifti,
#         )
#         dataloader = data.DataLoader(
#             _dataset,
#             batch_size = batch_size,
#             num_workers = num_workers,
#             shuffle = shuffle,
#             drop_last = drop_last,
#             pin_memory = pin_memory,
#         )
#         dataloaders.append(dataloader)
    
#     return dataloaders




    

