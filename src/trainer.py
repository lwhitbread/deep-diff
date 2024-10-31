import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import sys
import os
import time
from tqdm import tqdm 
from typing import Union, Iterable
import pandas as pd

sys.path.append("../")
from src.loss import *
from src.modules import *
from src.utils import * 

FloatIterable = Union[float, Iterable[float]]

class Trainer: 
    def __init__(
        self, 
        model: torch.nn.Module, 
        nb_dims: int,
        epochs: int, 
        loss_weights: dict,
        device: torch.device,
        dataset_args: dict,
        dls_and_template: list,
        loss_args: dict = None,
        use_ncc: bool = False,
        use_mixed_ncc_mse: bool = False,
        mixed_ncc_weight: float = 0.01,
        ncc_window: list = None,
        optimiser_type: str = "Adam",
        optimiser_args: dict = None,
        scheduler_type: str = "ReduceLROnPlateau",
        scheduler_args: dict = None,
        init_with_constant_sched: bool = False,
        constant_sched_epochs: int = 0,
        constant_sched_lr: float = 1e-4,
        mixed_precision: bool = False, 
        clip_grad: bool = True,
        clip_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        use_age_buckets: bool = False,
        run_monitor_args: int = None,
        model_save_freq: int = -1,
        enable_tqdm: bool = False,
        runtime_dir: str = None,
        intensity_field_multiplier_st_1: FloatIterable = None,
        intensity_field_multiplier_st_2: FloatIterable = None,
        freeze_intensity_st_1: int = 0,
        freeze_intensity_st_2: int = 0,
        nb_epochs_ignore_ct_unbiased_loss: int = 0,
        downsize_factor_int_st_1: float = 1,
        downsize_factor_int_st_2: float = 1,
        zero_mean_cons: bool = False,
        checkpoint: str = None,
        use_checkpoint: bool = False,
    ) -> None:
        
        self.model = model
        self.nb_dims = nb_dims
        assert self.nb_dims in [2, 3], "Invalid number of dimensions" 
        self.epochs = epochs
        self.loss_weights = loss_weights
        self.dataset_args = dataset_args
        self.dls_and_template = dls_and_template
        self.loss_args = loss_args
        self.use_ncc = use_ncc
        self.use_mixed_ncc_mse = use_mixed_ncc_mse
        self.mixed_ncc_weight = mixed_ncc_weight
        self.ncc_window = ncc_window
        self.device = device
        self.run_monitor_args = run_monitor_args
        self.clip_grad = clip_grad
        self.clip_grad_norm = clip_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_age_buckets = use_age_buckets
        self.best_val_loss = np.inf
        self.best_model_name = None
        self.save_best_model = False
        self.optimiser_type = optimiser_type
        assert self.optimiser_type in [
            "Adam", 
            "SGD", 
            "RMSprop", 
            "AdamW", 
            "Adamax", 
            "SparseAdam",
        ], "Invalid optimiser"
        self.optimiser_args = optimiser_args
        self.optimiser = None
        self.scheduler_type = scheduler_type
        assert self.scheduler_type in [
            "ReduceLROnPlateau", 
            "StepLR", 
            "MultiStepLR", 
            "ExponentialLR", 
            "CosineAnnealingLR", 
            "CosineAnnealingWarmRestarts",
            "CyclicLR",
            "OneCycleLR",
        ], "Invalid scheduler"
        self.scheduler_args = scheduler_args
        self.scheduler = None
        self.init_with_constant_sched = init_with_constant_sched
        self.constant_sched_epochs = constant_sched_epochs
        self.constant_sched_lr = constant_sched_lr
        self.len_train_loader = None
        self.model_save_freq = model_save_freq
        self.enable_tqdm = enable_tqdm
        self.mixed_precision = mixed_precision
        self.intensity_field_multiplier_st_1 = intensity_field_multiplier_st_1
        self.intensity_field_multiplier_st_2 = intensity_field_multiplier_st_2
        print(f"freeze_intensity_st_1 until epoch: {freeze_intensity_st_1}, freeze_intensity_st_2 until epoch: {freeze_intensity_st_2}")
        assert freeze_intensity_st_1 >= 0 and freeze_intensity_st_2 >= 0, \
            "freeze_intensity_st_1 and freeze_intensity_st_2 must be >= 0"
        assert freeze_intensity_st_1 <= self.epochs and freeze_intensity_st_2 <= self.epochs, \
            "freeze_intensity_st_1 and freeze_intensity_st_2 must be <= epochs"
        self.freeze_intensity_st_1 = freeze_intensity_st_1
        self.freeze_intensity_st_2 = freeze_intensity_st_2
        self.int_field_mult_fwd_st_1 = None
        self.int_field_mult_fwd_st_2 = None
        self.nb_epochs_ignore_ct_unbiased_loss = nb_epochs_ignore_ct_unbiased_loss
        self.downsize_factor_int_st_1 = downsize_factor_int_st_1
        self.downsize_factor_int_st_2 = downsize_factor_int_st_2
        self.runtime_dir = runtime_dir
        if self.runtime_dir is None:
            self.runtime_dir = os.getcwd()
        self.network_loop = self._network_loop
        self.network_execute = self._network_execute
        self.min_max_args = None
        self.zero_mean_cons = zero_mean_cons  
        self.checkpoint = checkpoint
        self.use_checkpoint = use_checkpoint
        self.checkpoint_loaded = False
        self.checkpoint_epoch = None

        try:
            self.img_dims = self.model.img_dims
        except:
            try:
                self.img_dims = self.model.module.img_dims
            except:
                raise Exception("Could not find image dimensions in model")
        
        self.prior_cond_temp_warp = None
        self.history = {
            "train": {
                "loss": [], 
                "recon_loss": [], 
                "grad_1_loss": [], 
                "disp_1_loss": [],
                "int_1_loss": [],
                "grad_2_loss": [],
                "disp_2_loss": [],
                "int_2_loss": [],
                "mean_tracker": [],
                "ct_unbiased_loss": [],
                "epoch": [],
                "time": [],
                "learning_rate": [],
            },
            "val": {
                "loss": [], 
                "recon_loss": [], 
                "grad_1_loss": [], 
                "disp_1_loss": [],
                "int_1_loss": [],
                "grad_2_loss": [],
                "disp_2_loss": [],
                "int_2_loss": [],
                "mean_tracker": [],
                "ct_unbiased_loss": [],
                "epoch": [],
                "time": [],
                "learning_rate": [],
            },
        }

    def set_dirs(
            self,
            checkpoint_setup: bool = False,
    ) -> None:
        if self.run_monitor_args is not None:
            _dir_path = f"{self.runtime_dir}/../models/{self.run_monitor_args['id']}"
            if not os.path.exists(_dir_path):
                os.makedirs(_dir_path)
            _len_dir_path = len([entry for entry in os.listdir(_dir_path) if os.path.isfile(os.path.join(_dir_path, entry))])
            if _len_dir_path > 0 and not checkpoint_setup:
                raise(FileExistsError(f"Directory {self.runtime_dir}/../models/{self.run_monitor_args['id']} \
                        is not empty. Please delete the contents of this directory before running the training."))
            _dir_path = f"{self.runtime_dir}/../results/{self.run_monitor_args['id']}/plots"
            self.dir_path_plots_data = f"{self.runtime_dir}/../results/{self.run_monitor_args['id']}/plots_data"
            if not os.path.exists(_dir_path):
                os.makedirs(_dir_path)
                os.makedirs(self.dir_path_plots_data)
            _len_dir_path = len([entry for entry in os.listdir(_dir_path) if os.path.isfile(os.path.join(_dir_path, entry))])
            if _len_dir_path > 0 and not checkpoint_setup:
                raise(FileExistsError(f"Directory {self.runtime_dir}/../results/{self.run_monitor_args['id']} \
                        is not empty. Please delete the contents of this directory before running the training."))
    
    def run(
        self
        ) -> None:
        
        start_train_time = time.time()
        
        dataloader_train = self.dls_and_template[0]
        self.len_train_loader = len(dataloader_train)
        dataloader_val = self.dls_and_template[1]
        self.min_max_args = self.dls_and_template[-2]
        self.volume_args = self.dls_and_template[-1]
        if len(self.dls_and_template) == 6:
            self.dataloader_test = self.dls_and_template[2]
            self.template = self.dls_and_template[3][0]
            self.template_seg = self.dls_and_template[3][1]
        else:
            self.template = self.dls_and_template[2][0]
            self.template_seg = self.dls_and_template[2][1]

        self.optimiser = getattr(
            optim, 
            self.optimiser_type
        )(
            self.model.parameters(), 
            **self.optimiser_args if self.optimiser_args else self.lr
        )
      
        default_scheduler_args = {
            "mode": "min",
            "factor": 0.5,
            "patience": 4,
            "verbose": True,
            "threshold": 1e-5,
            "threshold_mode": "rel",
            "cooldown": 2,
            "min_lr": 1e-5,
            "eps": 1e-08,
        }

        if self.scheduler_args is None:
            self.scheduler_args = default_scheduler_args
        
        self.scheduler = getattr(
            optim.lr_scheduler, 
            self.scheduler_type
        )(
            self.optimiser, 
            **self.scheduler_args
        )

        print(f"init with constant scheduler: {self.init_with_constant_sched}")
        if self.init_with_constant_sched:
            sched_lr_factor = self.constant_sched_lr / self.optimiser_args["lr"]
            print(f"Initialising with constant scheduler at {sched_lr_factor} of initial learning rate")
            self.init_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimiser,
                lr_lambda = lambda epoch: sched_lr_factor if epoch <= self.constant_sched_epochs else 0,
                verbose=True,
            )

        else:
            self.init_scheduler = None

        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        ## loss components
        self.losses = []
        self.weights = []
        
        # reconstruction loss
        if self.use_ncc:
            image_loss = NCC(self.device, win = self.ncc_window).loss
            print("Using NCC loss")
        elif self.use_mixed_ncc_mse:
            image_loss = MixedNCCMSE(
                self.device,
                ncc_weight = self.mixed_ncc_weight,
                win = self.ncc_window,
            ).loss
            print(f"Using mixed NCC and MSE loss with NCC weight {self.mixed_ncc_weight} and window {self.ncc_window}")
        else:
            image_loss = MSE().loss
            print("Using MSE loss")
        self.weights.append(self.loss_weights["recon"])
        self.losses.append(image_loss)

        # gradient loss on stage 1 displacement, order denotes derivative order
        grad_loss_cond_temp = Grad(penalty='l2', loss_mult = 1, order = 2, nb_dims = self.nb_dims).loss
        self.weights.append(self.loss_weights["grad_1"]) # default is 0.01
        self.losses.append(grad_loss_cond_temp)

        # deformation/displacement loss on stage 1
        # using this makes it difficult to morph the group template to an appropriate cond temp (i.e., matching analytic expectations)
        flow_loss_cond_temp = MSE().loss
        self.weights.append(self.loss_weights["disp_1"]) # default is 0.00
        self.losses.append(flow_loss_cond_temp)
        
        # gradient loss on stage 2 displacement, order denotes derivative order
        grad_loss = Grad(penalty='l2', loss_mult=1, order = 2, nb_dims = self.nb_dims).loss
        self.weights.append(self.loss_weights["grad_2"]) # default is 0.01
        self.losses.append(grad_loss)
        
        # deformation/displacement loss on stage 2
        flow_loss = MSE().loss
        self.weights.append(self.loss_weights["disp_2"]) # default is 0.01
        self.losses.append(flow_loss)

        # loss on the gen cond template intensity modulation field on stage 1
        cond_temp_intensity_loss = SparsePenalty(
            threshold = self.loss_args["SparsePenalty"]["threshold"], #if "threshold" in self.loss_args["SparsePenalty"] else 0.1, 
            nb_dims = self.nb_dims, 
            device = self.device,
            use_spatial_grad = self.loss_args["SparsePenalty"]["use_spatial_grad"], # if "use_spatial_grad" in self.loss_args["SparsePenalty"] else None,
            spatial_grad_mult = self.loss_args["SparsePenalty"]["spatial_grad_mult"], # if "spatial_grad_mult" in self.loss_args["SparsePenalty"] else 1.,
            downsample_factor = self.downsize_factor_int_st_1,
            img_dims = self.img_dims,
            ).loss
        self.weights.append(self.loss_weights["int_1"]) # default is 0.02
        self.losses.append(cond_temp_intensity_loss)

        # loss on the sample intensity modulation field on stage 2
        sample_intensity_loss = SparsePenalty(
            threshold = self.loss_args["SparsePenalty"]["threshold"], # if "threshold" in self.loss_args["SparsePenalty"] else 0.1, 
            nb_dims = self.nb_dims, 
            device = self.device,
            use_spatial_grad = self.loss_args["SparsePenalty"]["use_spatial_grad"], # if "use_spatial_grad" in self.loss_args["SparsePenalty"] else None,
            spatial_grad_mult = self.loss_args["SparsePenalty"]["spatial_grad_mult"], # if "spatial_grad_mult" in self.loss_args["SparsePenalty"] else 1.,
            downsample_factor = self.downsize_factor_int_st_2,
            img_dims = self.img_dims,
            ).loss
        self.weights.append(self.loss_weights["int_2"]) # default is 0.02
        self.losses.append(sample_intensity_loss)

        # loss on mean field where used to adjust cond templates
        mean_field_loss = MSE().loss
        # print(self.loss_weights["mean_trackers"])
        self.weights.append(self.loss_weights["mean_trackers"])
        self.losses.append(mean_field_loss)

        # L2 loss on cond temp less cond temp unbiased
        cond_temp_unbiased_loss = MSE().loss
        self.weights.append(self.loss_weights["cond_temp_unbiased"])
        self.losses.append(cond_temp_unbiased_loss)

        print(f"Loss weights for run: {self.weights}")

        self.epoch_step_time = []

        
        assert type(self.intensity_field_multiplier_st_1) == type(self.intensity_field_multiplier_st_2), \
            "intensity_field_multiplier_st_1 and intensity_field_multiplier_st_2 must be of same type. Either both scalar or both iterable or both None."
        
        freeze_int_weights_st_1 = False
        freeze_int_weights_st_2 = False
        
        print("Cuda available: ", torch.cuda.is_available())
        print("Device: ", self.device)
        if self.device.type == "cuda":
            print("No. of cuda devices: ", torch.cuda.device_count())
            print("Cuda memory allocated: ", torch.cuda.memory_allocated())

        for epoch in range(self.epochs):
            
            start = time.time()

            if self.use_checkpoint and not self.checkpoint_loaded and self.checkpoint is not None:
                self.load_checkpoint(self.checkpoint)
                self.checkpoint_loaded = True
                print(f"Loading checkpoint. Starting from epoch {self.checkpoint_epoch + 1}")
            
            if self.use_checkpoint and self.checkpoint_loaded:
                if epoch <= self.checkpoint_epoch:
                    print(f"Skipping epoch {epoch + 1}")
                    continue


            if self.intensity_field_multiplier_st_1 is not None and self.intensity_field_multiplier_st_2 is not None:

                if self.intensity_field_multiplier_st_1 == 0 and self.intensity_field_multiplier_st_2 == 0 and \
                    freeze_int_weights_st_1 and freeze_int_weights_st_2:
                        pass
                else:                
                    if isinstance(self.intensity_field_multiplier_st_1, (int, float)):
                        if self.intensity_field_multiplier_st_1 == 0 and self.intensity_field_multiplier_st_2 == 0:
                            freeze_int_weights_st_1 = True
                            freeze_int_weights_st_2 = True
                            self.weights[-4] = 0.
                            self.weights[-3] = 0.
                            print("Freezing intensity field weights and turning off intensity modulation for stage 1 and 2")

                            self.int_field_mult_fwd_st_1 = 0
                            self.int_field_mult_fwd_st_2 = 0

                        else:
                            if epoch >= self.freeze_intensity_st_1:
                                if freeze_int_weights_st_1:
                                    print("Stage 1 intensity field weights and intensity modulation are active")
                                freeze_int_weights_st_1 = False
                                self.weights[-4] = self.loss_weights["int_1"]
                                self.int_field_mult_fwd_st_1 = self.intensity_field_multiplier_st_1
                            else:
                                if not freeze_int_weights_st_1:
                                    print("Freezing stage 1 intensity field weights and turning off intensity modulation")
                                freeze_int_weights_st_1 = True
                                self.weights[-4] = 0.
                                self.int_field_mult_fwd_st_1 = 0.
                            
                            if epoch >= self.freeze_intensity_st_2:
                                if freeze_int_weights_st_2:
                                    print("Stage 2 intensity field weights and intensity modulation are active")
                                freeze_int_weights_st_2 = False
                                self.weights[-3] = self.loss_weights["int_2"]
                                self.int_field_mult_fwd_st_2 = self.intensity_field_multiplier_st_2
                            else:
                                if not freeze_int_weights_st_2:
                                    print("Freezing stage 2 intensity field weights and turning off intensity modulation")
                                freeze_int_weights_st_2 = True
                                self.weights[-3] = 0.
                                self.int_field_mult_fwd_st_2 = 0.

                    else:
                        try:
                            iter(self.intensity_field_multiplier_st_1)
                            iter(self.intensity_field_multiplier_st_2)
                            assert len(self.intensity_field_multiplier_st_1) == self.epochs
                            assert len(self.intensity_field_multiplier_st_2) == self.epochs
                        except:
                            raise(AssertionError("intensity_field_multiplier st 1 and st 2 variables must be a schedule of same length as epochs or a scalar."))
                        
                        if self.intensity_field_multiplier_st_1[epoch] == 0 and self.intensity_field_multiplier_st_2[epoch] == 0:
                            if not freeze_int_weights_st_1 and not freeze_int_weights_st_2:
                                print("Freezing intensity field weights and turning off intensity modulation")
                            freeze_int_weights_st_1 = True
                            freeze_int_weights_st_2 = True
                            self.weights[-4] = 0.
                            self.weights[-3] = 0.
                            self.int_field_mult_fwd_st_1 = 0.
                            self.int_field_mult_fwd_st_2 = 0.
                        
                        else:
                            if epoch >= self.freeze_intensity_st_1:
                                if freeze_int_weights_st_1:
                                    print("Stage 1 intensity field weights and intensity modulation are active")
                                freeze_int_weights_st_1 = False
                                self.weights[-4] = self.loss_weights["int_1"]
                                self.int_field_mult_fwd_st_1 = self.intensity_field_multiplier_st_1[epoch]
                            else:
                                if not freeze_int_weights_st_1:
                                    print("Freezing stage 1 intensity field weights and turning off intensity modulation")
                                freeze_int_weights_st_1 = True
                                self.weights[-4] = 0.
                                self.int_field_mult_fwd_st_1 = 0.
                            
                            if epoch >= self.freeze_intensity_st_2:
                                if freeze_int_weights_st_2:
                                    print("Stage 2 intensity field weights and intensity modulation are active")
                                freeze_int_weights_st_2 = False
                                self.weights[-3] = self.loss_weights["int_2"]
                                self.int_field_mult_fwd_st_2 = self.intensity_field_multiplier_st_2[epoch]
                            else:
                                if not freeze_int_weights_st_2:
                                    print("Freezing stage 2 intensity field weights and turning off intensity modulation")
                                freeze_int_weights_st_2 = True
                                self.weights[-3] = 0.
                                self.int_field_mult_fwd_st_2 = 0.        
                    
                    try:
                        self.model.freeze_intensity_weights(freeze_int_weights_st_1, freeze_int_weights_st_2)
                    except:
                        self.model.module.freeze_intensity_weights(freeze_int_weights_st_1, freeze_int_weights_st_2)

            if epoch == 0:
                self.set_dirs()
            if self.checkpoint_epoch is not None:
                if self.use_checkpoint and self.checkpoint_loaded:
                    if epoch == self.checkpoint_epoch + 1:
                        self.set_dirs(checkpoint_setup = True)
                     
            print(f"\nStarting epoch {epoch + 1}")

            self._do_epoch(
                epoch, 
                dataloader_train,
                epoch_type= "train",
            )
        
            val_loss = self._do_epoch(
                epoch, 
                dataloader_val,
                epoch_type= "val",
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.save_best_model = True
            
            self.save_history(
                best = self.save_best_model,
                last = True if epoch == self.epochs - 1 else False,
            )
            self.save_best_model = False

            if self.init_scheduler is not None and epoch < self.constant_sched_epochs:
                self.init_scheduler.step()
            else:
                if self.scheduler_type == "ReduceLROnPlateau":
                    self.scheduler.step(val_loss)
                
                elif self.scheduler_type in [
                    "StepLR", 
                    "MultiStepLR",
                    "ExponentialLR",
                    "CyclicLR",
                    "OneCycleLR",
                    "CosineAnnealingLR",    
                ]:
                    self.scheduler.step()
            
            if (epoch + 1) % self.model_save_freq == 0:
                if self.model_save_freq == -1: pass
                else:
                    if self.run_monitor_args is not None:
                        torch.save(self.model.state_dict(), f"{self.runtime_dir}/../models/{self.run_monitor_args['id']}/model_{epoch + 1}.pt")
            
            end = time.time()
            self.epoch_step_time.append(end - start)
            self.train_time = end - start_train_time
            print(f"  epoch {epoch + 1} took {end - start:.3f} seconds")

            
            if self.use_checkpoint:
                    self.save_checkpoint(
                        epoch = epoch,
                    )

        end_train_time = time.time()
        self.train_time = end_train_time - start_train_time
        print(f"\nTraining took {self.train_time / 60:.3f} minutes.")

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        
        print("Creating checkpoint")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "epoch": epoch,
            "train_time": self.train_time,
            "epoch_step_time": self.epoch_step_time,
        }
        
        # clean up old checkpoints
        for file in os.listdir(self.runtime_dir):
            if file.endswith(".pt") and "checkpoint" in file:
                os.remove(os.path.join(self.runtime_dir, file))
        
        if self.run_monitor_args is not None:
            torch.save(checkpoint, f"{self.runtime_dir}/../models/{self.run_monitor_args['id']}/run_{self.run_monitor_args['id']}_checkpoint_epoch_{epoch + 1}_time_{time.strftime('%Y-%m-%d_%H.%M.%S')}.pt")
        
        else:
            torch.save(checkpoint, f"{self.runtime_dir}/../models/checkpoint_epoch_{epoch + 1}_time_{time.strftime('%Y-%m-%d_%H.%M.%S')}.pt")
        

        print(f"Checkpoint saved at epoch {epoch + 1}")
        print(f"Training to this point took {self.train_time / 60:.3f} minutes")
        exit()
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
    ) -> None:
            
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.history = checkpoint["history"]
            self.best_val_loss = checkpoint["best_val_loss"]
            self.best_epoch = checkpoint["best_epoch"]
            self.train_time = checkpoint["train_time"]
            self.epoch_step_time = checkpoint["epoch_step_time"]
            self.checkpoint_loaded = True
            self.checkpoint_epoch = checkpoint["epoch"]
            print(f"Loaded checkpoint from {checkpoint_path}")

    
    def _do_epoch(
        self,
        epoch: int, 
        dataloader: torch.utils.data.DataLoader,
        epoch_type: str,
        ):

        assert epoch_type in ["train", "val"], "epoch_type must be either 'train' or 'val'"

        start_time = time.time()

        returns = self.network_loop(
            epoch,
            dataloader,
            epoch_type,
        )

        epoch_total_loss = returns[0]
        mean_epoch_loss = returns[1]
        int_dict = returns[2]
    
        
        self.history[epoch_type]["loss"].append(epoch_total_loss)
        self.history[epoch_type]["recon_loss"].append(mean_epoch_loss[0])
        self.history[epoch_type]["grad_1_loss"].append(mean_epoch_loss[1])
        self.history[epoch_type]["disp_1_loss"].append(mean_epoch_loss[2])
        self.history[epoch_type]["int_1_loss"].append(mean_epoch_loss[5])
        self.history[epoch_type]["grad_2_loss"].append(mean_epoch_loss[3])
        self.history[epoch_type]["disp_2_loss"].append(mean_epoch_loss[4])
        self.history[epoch_type]["int_2_loss"].append(mean_epoch_loss[6])
        self.history[epoch_type]["mean_tracker"].append(mean_epoch_loss[7])
        self.history[epoch_type]["ct_unbiased_loss"].append(mean_epoch_loss[8])
        self.history[epoch_type]["epoch"].append(epoch + 1)
        self.history[epoch_type]["time"].append(time.time() - start_time)
        self.history[epoch_type]["learning_rate"].append(self.optimiser.param_groups[0]['lr'] if epoch_type == "train" else None)

        print(f"  {epoch_type} loss (total loss): {epoch_total_loss}")
        
        int_1_max = int_dict["int_1_max"]
        int_1_min = int_dict["int_1_min"]
        int_2_max = int_dict["int_2_max"]
        int_2_min = int_dict["int_2_min"]
        int_1_mean = int_dict["int_1_mean"]
        int_2_mean = int_dict["int_2_mean"]

        # TODO: temporary printout while prototyping
        if self.intensity_field_multiplier_st_1 != 0 or self.intensity_field_multiplier_st_2 != 0:
            print(f"\nIntensity field stats for this {epoch_type} epoch {epoch + 1}:")
            print(f"1 intensity field max: {int_1_max:.6f}")
            print(f"1 intensity field min: {int_1_min:.6f}")
            print(f"1 intensity field mean: {int_1_mean:.6f}")
            print(f"2 intensity field max: {int_2_max:.6f}")
            print(f"2 intensity field min: {int_2_min:.6f}")
            print(f"2 intensity field mean: {int_2_mean:.6f}")
            print()
        
        if (epoch + 1) % 5 == 0:
            print(f"  {epoch_type} loss (elements): \n \
                    \tgrad_cond_temp: {mean_epoch_loss[1]}; \n \
                    \tflow_cond_temp: {mean_epoch_loss[2]}; \n \
                    \tintensity_cond_temp: {mean_epoch_loss[5]}; \n \
                    \tgrad_sample: {mean_epoch_loss[3]}; \n \
                    \tflow_sample: {mean_epoch_loss[4]}; \n \
                    \tintensity_sample: {mean_epoch_loss[6]}; \n \
                    \tmean_tracker: {mean_epoch_loss[7]}; \n \
                    \tct_unbiased: {mean_epoch_loss[8]}; \n \
                    \treconstruct: {mean_epoch_loss[0]} \n" 
            )

        if epoch_type == "train":
            del returns
        if epoch_type == "val":
            del returns
            return epoch_total_loss

    def _network_loop(
        self,
        epoch: int,
        dataloader: data.DataLoader,
        epoch_type: str,
    ):
        epoch_loss = []
        epoch_total_loss = []

        self.optimiser.zero_grad(set_to_none = True)
        
        self.model.train() if epoch_type == "train" else self.model.eval()
        training = True if epoch_type == "train" else False

        len_dataloader = len(dataloader)
        
        int_1_max = -np.inf
        int_1_min = np.inf
        int_1_mean = 0.
        int_2_max = -np.inf
        int_2_min = np.inf
        int_2_mean = 0.
        
        for idx_0, batch in tqdm(
            enumerate(dataloader), 
            total = len_dataloader, 
            desc=f"Epoch {epoch + 1} {epoch_type}", 
            leave = False,
            disable = not self.enable_tqdm,
        ):
            
            images = batch["image"]
    
            scan_path = batch["scan_path"]
            curr_batch_size = images.shape[0]

            if self.use_age_buckets:
                params = batch["age_bucket"].view(curr_batch_size, 1).float().to(self.device)
                if idx_0 == 0 and epoch == 0:
                    print(f"using age buckets: {params}")
            else:
                params = batch["param"].view(curr_batch_size, 1).float()
                params = params.to(self.device)

                if idx_0 == 0 and epoch == 0:
                    print(f"using normal ages: {params.view(-1)}")
            
            images = images.to(self.device)

            dim_tile = [1 for _ in range(self.nb_dims)]
            template_tensor = torch.tile(
                self.template, 
                (curr_batch_size, 1, *dim_tile),
            ).float().to(self.device)
            
            zeros = torch.Tensor([0.]).to(self.device)

            means_idx_0 = batch["means_idx_0"].to(self.device)
            prop_means_idx_0 = batch["prop_means_idx_0"].to(torch.float32).to(self.device)
            assert means_idx_0.shape[0] == curr_batch_size

            lost_list, loss, pred = self.network_execute(
                params,
                template_tensor,
                images,
                zeros,
                training,
                epoch,
                epoch_type,
                idx_0,
                means_idx_0,
                prop_means_idx_0,
            )

            epoch_loss.append(lost_list)
            epoch_total_loss.append(sum(lost_list))
                
            _temp_int_1_max = pred[3].detach().cpu().max()
            int_1_max = _temp_int_1_max if _temp_int_1_max > int_1_max else int_1_max    
            _temp_int_1_min = pred[3].detach().cpu().min()
            int_1_min = _temp_int_1_min if _temp_int_1_min < int_1_min else int_1_min  
            _temp_int_2_max = pred[4].detach().cpu().max()
            int_2_max = _temp_int_2_max if _temp_int_2_max > int_2_max else int_2_max
            _temp_int_2_min = pred[4].detach().cpu().min()
            int_2_min = _temp_int_2_min if _temp_int_2_min < int_2_min else int_2_min         
            
            _int_1_mean = pred[3].detach().cpu().mean()
            _int_2_mean = pred[4].detach().cpu().mean()

            curr_mean_prop = 1. / (idx_0 + 1)
            old_mean_prop = 1. - curr_mean_prop
            int_1_mean = _int_1_mean * curr_mean_prop + int_1_mean * old_mean_prop
            int_2_mean = _int_2_mean * curr_mean_prop + int_2_mean * old_mean_prop

        int_dict = {
            "int_1_max": int_1_max,
            "int_1_min": int_1_min,
            "int_2_max": int_2_max,
            "int_2_min": int_2_min,
            "int_1_mean": int_1_mean,
            "int_2_mean": int_2_mean,
        }
                
        images = None
        template_tensor = None
        params = None
        means_idx_0 = None
        prop_means_idx_0 = None
        pred = None

        epoch_total_loss = np.mean(epoch_total_loss)
        mean_epoch_loss = np.mean(epoch_loss, axis = 0)

        return (
            epoch_total_loss, 
            mean_epoch_loss, 
            int_dict, 
        )
    
    def _network_execute(
        self,
        params: torch.Tensor,
        template_tensor: torch.Tensor,
        images: torch.Tensor,
        zeros: torch.Tensor,
        training: bool,
        epoch: int,
        epoch_type: str,
        itr: int,
        means_idx_0: torch.Tensor,
        prop_means_idx_0: torch.Tensor,
    ):
        
        with torch.autocast(
            enabled = self.mixed_precision, 
            dtype = torch.float16 if self.mixed_precision else torch.float32,
            device_type = "cuda" if torch.cuda.is_available() else "cpu",
        ):
            if epoch_type == "train":
                if not self.zero_mean_cons:
                    pred = self.model(
                        params, 
                        template_tensor, 
                        images, 
                        training = training, 
                        intensity_field_multiplier_st_1 = self.int_field_mult_fwd_st_1,
                        intensity_field_multiplier_st_2 = self.int_field_mult_fwd_st_2,
                    )
                else:
                    pred = self.model(
                        params, 
                        template_tensor, 
                        images, 
                        training = training, 
                        intensity_field_multiplier_st_1 = self.int_field_mult_fwd_st_1,
                        intensity_field_multiplier_st_2 = self.int_field_mult_fwd_st_2,
                        means_idx_0 = means_idx_0,
                        prop_means_idx_0 = prop_means_idx_0,
                    )
            else:
                with torch.no_grad():
                    if not self.zero_mean_cons:
                        pred = self.model(
                            params, 
                            template_tensor, 
                            images, 
                            training = training, 
                            intensity_field_multiplier_st_1 = self.int_field_mult_fwd_st_1,
                            intensity_field_multiplier_st_2 = self.int_field_mult_fwd_st_2,
                        )

                    else:
                        pred = self.model(
                            params, 
                            template_tensor, 
                            images, 
                            training = training, 
                            intensity_field_multiplier_st_1 = self.int_field_mult_fwd_st_1,
                            intensity_field_multiplier_st_2 = self.int_field_mult_fwd_st_2,
                            means_idx_0 = means_idx_0,
                            prop_means_idx_0 = prop_means_idx_0,
                        )

            loss = 0.
            lost_list = []
            
            for idx_1, loss_fn in enumerate(self.losses):
                y_true = images if idx_1 == 0 else zeros
                y_true = pred[5] if idx_1 == 5 else y_true # here y_true is the template tensor
                y_true = pred[6] if idx_1 == 6 else y_true # here y_true is the cond_temp ####
                y_true = pred[5] if idx_1 == 8 else y_true # here y_true is the predicted image
                if idx_1 == 0:
                    y_pred = pred[0]
                elif idx_1 < 3:
                    y_pred = pred[1]
                elif idx_1 < 5:
                    y_pred = pred[2]
                elif idx_1 == 5:
                    y_pred = pred[3] 
                elif idx_1 == 6:
                    y_pred = pred[4] 
                elif idx_1 == 7:
                    y_pred = pred[-4] if self.zero_mean_cons else zeros
                elif idx_1 == 8:
                    y_pred = pred[8] if self.zero_mean_cons else zeros
                else:
                    raise ValueError("loss index out of range")

                # only apply cortical mask to recon loss
                if idx_1 == 8 and epoch < self.nb_epochs_ignore_ct_unbiased_loss:
                    curr_loss = 0. * loss_fn(y_true, y_pred)
                else:
                    curr_loss = self.weights[idx_1] * loss_fn(y_true, y_pred)
                
                loss += curr_loss
                lost_list.append(curr_loss.item())
     
        if epoch_type == "train":
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else: 
                loss.backward()
            
            if (itr + 1) % self.gradient_accumulation_steps == 0:
                if self.clip_grad:
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimiser)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                if self.mixed_precision:
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                else:
                    self.optimiser.step()
                if self.init_scheduler is not None and epoch < self.constant_sched_epochs:
                    pass # scheduler step is taken in the run function
                else:
                    if self.scheduler_type == "CosineAnnealingWarmRestarts":
                        self.scheduler.step(epoch + itr / self.len_train_loader)
                self.optimiser.zero_grad(set_to_none = True)

        return lost_list, loss, pred
    
    def save_history(
        self,
        best: bool = False,
        last: bool = False,
    ) -> None:
        """
        Saves the history of the training and validation losses
        """

        if best:
            if self.best_model_name is not None:
                os.system(f"rm {self.runtime_dir}/../models/{self.run_monitor_args['id']}/{self.best_model_name}")
            
            self.best_model_name = f"best_model_run_{self.run_monitor_args['id']}_epoch_{self.best_epoch}_val_loss_{self.best_val_loss:.6f}_time_{time.strftime('%Y-%m-%d_%H.%M.%S')}.pt"
            torch.save(
                self.model.state_dict(),
                f"{self.runtime_dir}/../models/{self.run_monitor_args['id']}/{self.best_model_name}",
            )

        if last:
            torch.save(
                self.model.state_dict(), 
                f"{self.runtime_dir}/../models/{self.run_monitor_args['id']}/final_model_run_{self.run_monitor_args['id']}.pt",
            )

        _df_train = pd.DataFrame(self.history["train"])
        _df_val = pd.DataFrame(self.history["val"])
        _df_train.columns = ["train_" + col for col in _df_train.columns]
        _df_val.columns = ["val_" + col for col in _df_val.columns]
        _df = pd.concat([_df_train, _df_val], axis = 1)
        _df.to_csv(f"{self.runtime_dir}/../results/{self.run_monitor_args['id']}/history_run_{self.run_monitor_args['id']}.csv", index = False)