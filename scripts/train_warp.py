import torch
import os
import argparse
import configparser
import datetime

# sys.path.append("../")
from src.loss import *
from src.modules import *
from src.networks import *
from src.datasets import *
from src.trainer import *

if __name__ == "__main__":
# Get system arguments to apply through script
    parser = argparse.ArgumentParser() 
    parser.add_argument("--config-file", required=True, help="config filename to use")
    parser.add_argument("--run-id", required=True, help="run id to use for saving results")
    parser.add_argument("--device", required=False, help="device to use for training", default = "0")
    parser.add_argument("--checkpoint", required=False, help="path to checkpoint", default = None)
    parser.add_argument("--use-checkpoint", required=False, help="use checkpoint", default = False)
    args = parser.parse_args()
    config_file = args.config_file
    device = int(args.device)
    use_checkpoint = eval(args.use_checkpoint)
    checkpoint = args.checkpoint
    
    run_id = args.run_id
    assert run_id != "", "run_id must not be empty"

    config_parent = "../config_files/"
    config_file_path = config_parent + config_file
    print(f"Config file: {config_file_path}")
    print(f"Run ID: {run_id}")

    # time stamp
    now = datetime.datetime.utcnow()
    time_stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Time stamp: {time_stamp} UTC")

    # Read config file
    config = configparser.ConfigParser() 
    config.read(config_file_path)

    nb_dims = int(config["run_args"]["nb_dims"])
    epochs = int(config["run_args"]["epochs"])
    mixed_precision = eval(config["run_args"]["mixed_precision"])
    clip_grad = eval(config["run_args"]["clip_grad"])
    clip_grad_norm = float(config["run_args"]["clip_grad_norm"])
    gradient_accumulation_steps = int(config["run_args"]["gradient_accumulation_steps"])
    use_age_buckets = eval(config["run_args"]["use_age_buckets"])
    model_save_freq = int(config["run_args"]["model_save_freq"])
    enable_tqdm = eval(config["run_args"]["enable_tqdm"])
    intensity_field_multiplier_st_1 = float(config["run_args"]["intensity_field_multiplier_st_1"])
    intensity_field_multiplier_st_2 = float(config["run_args"]["intensity_field_multiplier_st_2"])
    freeze_intensity_st_1 = int(config["run_args"]["freeze_intensity_st_1"])
    freeze_intensity_st_2 = int(config["run_args"]["freeze_intensity_st_2"])
    nb_epochs_ignore_ct_unbiased_loss = int(config["run_args"]["nb_epochs_ignore_ct_unbiased_loss"])
    optimiser = config["run_args"]["optimiser"]
    optimiser_args = eval(config["run_args"]["optimiser_args"])
    scheduler = config["run_args"]["scheduler"]
    scheduler_args = eval(config["run_args"]["scheduler_args"])
    init_with_constant_sched = eval(config["run_args"]["init_with_constant_sched"])
    constant_sched_epochs = int(config["run_args"]["constant_sched_epochs"])
    constant_sched_lr = float(config["run_args"]["constant_sched_lr"])
    
    dataset_args = eval(config["dataset_args"]["dataset_args"])
    
    dataset_root = config[f"data_paths"]["dataset_root"]
    template_path = config[f"data_paths"]["template_path"]

    print(f"Dataset root: {dataset_root}")
    print(f"Template path: {template_path}")

    runtime_dir = os.getcwd()
    
    try:
        template_avg_int_shift_factor = dataset_args["template_avg_int_shift_factor"]
        assert isinstance(template_avg_int_shift_factor, float), "template_avg_int_shift_factor must be a float"
        assert template_avg_int_shift_factor >= 0.9 and template_avg_int_shift_factor <=1.1, "template_avg_int_shift_factor must be >= 0.9 and <= 1.1"
        print(f"Template_avg_int_shift_factor: {template_avg_int_shift_factor}")
    except:
        print("No template_avg_int_shift_factor found in dataset_args, so default is no average intensity shift to the template")

    dataset_args.update(
        template_path = template_path,
        dataset_root = dataset_root,
        dataset_file_path = f"{dataset_root}/dataset.csv",
        runtime_dir = runtime_dir,
        run_id = run_id,
    )

    try:
        _loader_file_path_0 = dataset_args["loader_file_paths"][0]
        if "female" in _loader_file_path_0:
            dataset_args.update(
                dataset_file_path = f"{dataset_root}/dataset_female.csv",
            )
        elif "male" in _loader_file_path_0:
            dataset_args.update(
                dataset_file_path = f"{dataset_root}/dataset_male.csv",
            )
    except:
        pass

    if dataset_args["dtype"] != "float32":
        assert mixed_precision, "dtype must be float32 if the <mixed_precision> flag is False"

    nb_unet_feats_reg = eval(config["model_args"]["nb_unet_feats_reg"])

    try:
        unet_upsample_mode = config["model_args"]["unet_upsample_mode"]
        print(f"unet_upsample_mode: {unet_upsample_mode}")
    except:
        unet_upsample_mode = "nearest"
        print(f"No unet_upsample_mode found in config file, so default is nearest")

    nb_unet_feats_int  = eval(config["model_args"]["nb_unet_feats_int"])
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    img_dims = eval(config["model_args"]["img_dims"])
    gen_features = int(config["model_args"]["gen_features"])
    gen_features_blocks = int(config["model_args"]["gen_features_blocks"])
    nb_feature_streams = int(config["model_args"]["nb_feature_streams"])
    int_steps = int(config["model_args"]["int_steps"])
    interp_mode = config["model_args"]["interp_mode"]
    create_field_nb_layers = int(config["model_args"]["create_field_nb_layers"])
    
    try:
        int_smoothing_args_st_1 = eval(config["model_args"]["int_smoothing_args_st_1"])
        int_smoothing_args_st_2 = eval(config["model_args"]["int_smoothing_args_st_2"])
        print(f"int_smoothing_args_st_1: {int_smoothing_args_st_1}")
        print(f"int_smoothing_args_st_2: {int_smoothing_args_st_2}")
    except:
        print("No int_smoothing_args_st_1 or int_smoothing_args_st_2 found in config file, so trying to use int_smoothing_args")
        int_smoothing_args_st_1 = eval(config["model_args"]["int_smoothing_args"])
        int_smoothing_args_st_2 = eval(config["model_args"]["int_smoothing_args"])
        print(f"int_smoothing_args_st_1: {int_smoothing_args_st_1}")
        print(f"int_smoothing_args_st_2: {int_smoothing_args_st_2}")
    
    # int_smoothing_args = eval(config["model_args"]["int_smoothing_args"])
    int_smoothing_args_mean_trackers = None
    try:
        int_smoothing_args_mean_trackers = eval(config["model_args"]["int_smoothing_args_mean_trackers"])
        print(f"int_smoothing_args_mean_trackers: {int_smoothing_args_mean_trackers}")
    except:
        print("No int_smoothing_args_mean_trackers found in config file, so default is None\n \
              (i.e., use int_smoothing_args for mean trackers)")
        pass
    intensity_field_act = config["model_args"]["intensity_field_act"]
    intensity_field_act_mult = float(config["model_args"]["intensity_field_act_mult"])
    intensity_act = config["model_args"]["intensity_act"]
    downsize_factor_int_st_1 = float(config["model_args"]["downsize_factor_int_st_1"])
    downsize_factor_int_st_2 = float(config["model_args"]["downsize_factor_int_st_2"])
    downsize_factor_vec_st_1 = float(config["model_args"]["downsize_factor_vec_st_1"])
    downsize_factor_vec_st_2 = float(config["model_args"]["downsize_factor_vec_st_2"])
    use_tanh_vecint_act = eval(config["model_args"]["use_tanh_vecint_act"])
    zero_mean_cons = eval(config["model_args"]["zero_mean_cons"])
    adj_warps = eval(config["model_args"]["adj_warps"])
    lambda_rate = float(config["model_args"]["lambda_rate"])
    compile_flag = None
    try:
        compile_flag = eval(config["model_args"]["compile_flag"])
    except:
        pass
    
    dls_and_template = get_dls_and_template(
        **dataset_args,
    )
    min_max_args = dls_and_template[-2]

    model = Warp(
        nb_dims = nb_dims,
        gen_features = gen_features, 
        img_dims = img_dims, 
        gen_features_blocks = gen_features_blocks,
        nb_gen_features_streams = nb_feature_streams,
        int_steps = int_steps, 
        interp_mode = interp_mode, 
        nb_unet_features_reg = nb_unet_feats_reg,
        unet_upsample_mode = unet_upsample_mode,
        nb_unet_features_int = nb_unet_feats_int,
        device = device,
        int_smoothing_args_st_1 = int_smoothing_args_st_1,
        int_smoothing_args_st_2 = int_smoothing_args_st_2,
        int_smoothing_args_mean_trackers = int_smoothing_args_mean_trackers if int_smoothing_args_mean_trackers is not None else None,
        intensity_field_act = intensity_field_act,
        intensity_field_act_mult = intensity_field_act_mult,
        intensity_act = intensity_act,
        downsize_factor_intensity_st_1 = downsize_factor_int_st_1,
        downsize_factor_intensity_st_2 = downsize_factor_int_st_2,
        downsize_factor_vec_st_1 = downsize_factor_vec_st_1,
        downsize_factor_vec_st_2 = downsize_factor_vec_st_2,
        use_tanh_vecint_act = use_tanh_vecint_act,
        zero_mean_cons = zero_mean_cons,
        adj_warps = adj_warps,
        lambda_rate = lambda_rate,
        min_max_args = min_max_args,
        nb_field_layers = create_field_nb_layers,
    )

    loss_args = eval(config["loss_args"]["loss_args"])
    loss_weights = eval(config["loss_args"]["loss_weights"])
    try:
        use_ncc = eval(config["loss_args"]["use_ncc"])
    except:
        use_ncc = False
    try:
        use_mixed_ncc_mse = eval(config["loss_args"]["use_mixed_ncc_mse"])
        mixed_ncc_weight = float(config["loss_args"]["mixed_ncc_weight"])
    except:
        use_mixed_ncc_mse = False
        mixed_ncc_weight = None
    try:
        ncc_window = eval(config["loss_args"]["ncc_window"])
    except:
        ncc_window = None
    
    assert use_ncc == False or use_mixed_ncc_mse == False, \
        "use_ncc and use_mixed_ncc_mse cannot both be True"
    
    if use_ncc or use_mixed_ncc_mse:
        assert ncc_window is not None, \
            "ncc_window must be provided if use_ncc or use_mixed_ncc_mse is True"

    run_monitor_args = dict(
        project = "conditional_warp",
        name = f"run_{run_id}",
        id = str(run_id),
    )

    model.summary_mode(summary_mode = False)
    model = model.to(device)

    if torch.__version__.split(".")[0] == "2":
        if compile_flag is None:   
            compile_flag = False
        if compile_flag:    
            print("Attempting to Compiling model...")
            model = torch.compile(model = model)
            torch.set_float32_matmul_precision('high')
            torch._dynamo.config.verbose = True
            torch._dynamo.config.cache_size_limit = 2560
            print("Compiling model complete")
    else:
        compile_flag = False

    trainer = Trainer(
        model,
        nb_dims,
        epochs,
        loss_weights,
        device,
        dataset_args = dataset_args,
        dls_and_template = dls_and_template,
        loss_args = loss_args,
        use_ncc = use_ncc,
        use_mixed_ncc_mse = use_mixed_ncc_mse,
        mixed_ncc_weight = mixed_ncc_weight,
        ncc_window = ncc_window,
        optimiser_type = optimiser,
        optimiser_args = optimiser_args,
        scheduler_type = scheduler,
        scheduler_args = scheduler_args,
        init_with_constant_sched = init_with_constant_sched,
        constant_sched_epochs = constant_sched_epochs,
        constant_sched_lr = constant_sched_lr,
        mixed_precision = mixed_precision,
        clip_grad = clip_grad,
        clip_grad_norm = clip_grad_norm,
        gradient_accumulation_steps = gradient_accumulation_steps,
        use_age_buckets = use_age_buckets,
        run_monitor_args = run_monitor_args,
        model_save_freq = model_save_freq,
        enable_tqdm = enable_tqdm,
        runtime_dir = runtime_dir,
        intensity_field_multiplier_st_1 = intensity_field_multiplier_st_1,
        intensity_field_multiplier_st_2 = intensity_field_multiplier_st_2,
        freeze_intensity_st_1 = freeze_intensity_st_1,
        freeze_intensity_st_2 = freeze_intensity_st_2,
        nb_epochs_ignore_ct_unbiased_loss = nb_epochs_ignore_ct_unbiased_loss,
        downsize_factor_int_st_1 = downsize_factor_int_st_1,
        downsize_factor_int_st_2 = downsize_factor_int_st_2,
        zero_mean_cons = zero_mean_cons,
        checkpoint = checkpoint,
        use_checkpoint = use_checkpoint,
    )

    trainer.run()