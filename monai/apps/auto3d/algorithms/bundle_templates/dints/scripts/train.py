# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import json
import logging
import math
import monai
import numpy as np
import os
import sys
import time
import torch
import torch.distributed as dist
import yaml

from datetime import datetime
from monai.data import (
    partition_dataset,
    CacheDataset,
    ThreadDataLoader,
)
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai.transforms import (
    apply_transform,
    Randomizable,
    Transform,
    AsDiscrete,
    CastToTyped,
    Compose,
    CopyItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByLabelClassesd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    Spacingd,
    SpatialPadd,
)
from monai.utils import set_determinism
from scipy import ndimage
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter


class DuplicateCacheDataset(CacheDataset):
    def __init__(self, times: int, **kwargs):
        super().__init__(**kwargs)
        self.times = times

    def __len__(self):
        return self.times * super().__len__()

    def _transform(self, index: int):
        # print("index", index)
        index = index // self.times
        if index % len(self) >= self.cache_num:  # support negative index
            # no cache for this index, execute all the transforms directly
            return super()._transform(index)
        # load data from cache and execute from the first random transform
        start_run = False
        if self._cache is None:
            self._cache = self._fill_cache()
        data = self._cache[index]
        if not isinstance(self.transform, Compose):
            raise ValueError(
                "transform must be an instance of monai.transforms.Compose."
            )
        for _transform in self.transform.transforms:
            if (
                start_run
                or isinstance(_transform, Randomizable)
                or not isinstance(_transform, Transform)
            ):
                # only need to deep copy data on first non-deterministic transform
                if not start_run:
                    start_run = True
                    data = copy.deepcopy(data)
                data = apply_transform(_transform, data)
        return data

    def __item__(self, index: int):
        return super().__item__(index // self.times)


def main():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument(
        "--arch_ckpt",
        action="store",
        required=True,
        help="data root",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint full path",
    )
    parser.add_argument(
        "--fold",
        action="store",
        required=True,
        help="fold index in N-fold cross-validation",
    )
    parser.add_argument(
        "--input_info",
        action="store",
        required=True,
        help="input information",
    )
    parser.add_argument(
        "--json_key",
        action="store",
        required=True,
        help="selected key in .json data list",
    )
    parser.add_argument(
        "--local_rank",
        required=int,
        help="local process rank",
    )
    parser.add_argument(
        "--num_folds",
        action="store",
        required=True,
        help="number of folds in cross-validation",
    )
    parser.add_argument(
        "--output_root",
        action="store",
        required=True,
        help="output root",
    )
    parser.add_argument(
        "--repo_root",
        action="store",
        required=True,
        help="repository root",
    )
    parser.add_argument(
        "--data_stat",
        action="store",
        required=True,
        help="data stat",
    )
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root, exist_ok=True)

    amp = True
    determ = False
    fold = int(args.fold)
    input_channels = None
    learning_rate = 0.025
    learning_rate_milestones = np.array([0.2, 0.4, 0.6, 0.8])
    num_images_per_batch = 2
    num_epochs = None
    num_epochs_per_validation = None
    num_folds = int(args.num_folds)
    num_patches_per_image = 1
    num_sw_batch_size = 6
    output_classes = None
    overlap_ratio = 0.625
    patch_size = [96, 96, 96]
    patch_size_valid = patch_size
    spacing = [1.0, 1.0, 1.0]

    # deterministic training
    if determ:
        set_determinism(seed=0)

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")

    # setting for different datasets
    with open(args.data_stat) as f_data_stat:
        data_stat = yaml.full_load(f_data_stat)

    with open(args.input_info) as f_input_info:
        input_info = yaml.full_load(f_input_info)

    data_root = input_info["dataroot"]
    input_channels = int(data_stat["stats_summary"]["image_stats"]["channels"]["max"])
    output_classes = len(data_stat["stats_summary"]["label_stats"]["labels"])
    max_shape = data_stat["stats_summary"]["image_stats"]["shape"]["max"][0]
    for _k in range(3):
        patch_size[_k] = max(32, max_shape[_k] // 32 * 32) if max_shape[_k] < patch_size[_k] else patch_size[_k]
    patch_size_valid = patch_size

    nomalizing_transform = None
    if input_info["modality"].lower() == "ct":
        intensity_upper_bound = float(
            data_stat["stats_summary"]["image_foreground_stats"]["intensity"][
                "percentile_99_5"
            ][0]
        )
        intensity_lower_bound = float(
            data_stat["stats_summary"]["image_foreground_stats"]["intensity"][
                "percentile_00_5"
            ][0]
        )
        if dist.get_rank() == 0:
            print("[info] intensity_upper_bound", intensity_upper_bound)
            print("[info] intensity_lower_bound", intensity_lower_bound)
        nomalizing_transform = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=intensity_lower_bound,
                a_max=intensity_upper_bound,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    else:
        spacing = data_stat["stats_summary"]["image_stats"]["spacing"]["median"]

        nomalizing_transform = [
            NormalizeIntensityd(
                keys=["image"], nonzero=True, channel_wise=True
            ),
        ]

    if dist.get_rank() == 0:
        print("[info] image modality:", input_info["modality"])
        print("[info] input_channels", input_channels)
        print("[info] output_classes", output_classes)
        print("[info] patch_size", patch_size)
        print("[info] patch_size_valid", patch_size_valid)

    dist.barrier()
    world_size = dist.get_world_size()

    # load data list (.json)
    with open(os.path.join(args.repo_root, input_info["datalist"])) as f:
        json_data = json.load(f)

    list_train = []
    list_valid = []
    for item in json_data[args.json_key]:
        if item["fold"] == fold:
            item.pop("fold", None)
            list_valid.append(item)
        else:
            item.pop("fold", None)
            list_train.append(item)

    # training data
    files = []
    for _i in range(len(list_train)):
        str_img = os.path.join(data_root, list_train[_i]["image"])
        str_seg = os.path.join(data_root, list_train[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})

    train_files = files

    num_iterations_per_validation = 500
    num_validation_rounds = 80
    if len(train_files) > 60:
        num_epochs_per_validation = math.ceil(
            float(len(train_files)) / float(world_size)
        )
        num_epochs_per_validation = math.ceil(
            num_epochs_per_validation / float(num_images_per_batch)
        )
        num_epochs_per_validation = math.ceil(
            float(num_iterations_per_validation) / num_epochs_per_validation
        )
        train_files = partition_dataset(
            data=train_files,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=True,
        )[dist.get_rank()]
    else:
        num_epochs_per_validation = math.ceil(
            float(len(train_files)) / float(num_images_per_batch)
        )
        num_epochs_per_validation = math.ceil(
            float(num_iterations_per_validation) / num_epochs_per_validation
        )
    num_epochs = num_epochs_per_validation * num_validation_rounds

    if dist.get_rank() == 0:
        print("train_files:", len(train_files))
        print("num_epochs_per_validation:", num_epochs_per_validation)
        print("num_validation_rounds:", num_validation_rounds)

    import random

    random.shuffle(train_files)

    # validation data
    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(data_root, list_valid[_i]["image"])
        str_seg = os.path.join(data_root, list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    val_files = files
    if len(val_files) < world_size:
        val_files = val_files * math.ceil(float(world_size) / float(len(val_files)))
    val_files = partition_dataset(
        data=val_files, shuffle=False, num_partitions=world_size, even_divisible=False
    )[dist.get_rank()]

    if dist.get_rank() == 0:
        print("val_files:", len(val_files))

    # network architecture
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
                align_corners=(True, True),
            ),
            CastToTyped(keys="image", dtype=torch.float32),
        ]
        + nomalizing_transform
        + [
            CopyItemsd(keys=["label"], times=1, names=["label4crop"]),
            Lambdad(
                keys=["label4crop"],
                func=lambda x: np.concatenate(
                    tuple(

                            ndimage.binary_dilation(
                                (x == _k).astype(x.dtype),
                                iterations=max(patch_size) // 2,
                            ).astype(x.dtype)
                            for _k in range(output_classes)

                    ),
                    axis=0,
                ),
                overwrite=True,
            ),
            EnsureTyped(keys=["image", "label"]),
            CastToTyped(keys=["image"], dtype=(torch.float16)),
            RandShiftIntensityd(keys=["image"], offsets=0.0, prob=0.0),
            CastToTyped(keys=["image"], dtype=(torch.float32)),
            SpatialPadd(
                keys=["image", "label", "label4crop"],
                spatial_size=patch_size,
                mode=["reflect", "constant", "constant"],
            ),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label4crop",
                num_classes=output_classes,
                ratios=[
                    1,
                ]
                * output_classes,
                spatial_size=patch_size,
                num_samples=num_patches_per_image,
            ),
            Lambdad(keys=["label4crop"], func=lambda x: 0),
            RandRotated(
                keys=["image", "label"],
                range_x=0.3,
                range_y=0.3,
                range_z=0.3,
                mode=["bilinear", "nearest"],
                prob=0.2,
            ),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.8,
                max_zoom=1.2,
                mode=["trilinear", "nearest"],
                align_corners=[True, None],
                prob=0.16,
            ),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.15,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
                align_corners=(True, True),
            ),
            EnsureTyped(keys=["image", "label"]),
            CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.uint8)),
        ]
        + nomalizing_transform
        + [
            CastToTyped(keys=["image"], dtype=(torch.float16)),
            RandShiftIntensityd(keys=["image"], offsets=0.0, prob=0.0),
            CastToTyped(keys=["image"], dtype=(torch.float32)),
        ]
    )

    train_ds = DuplicateCacheDataset(
        cache_rate=1.0,
        data=train_files,
        num_workers=8,
        times=num_epochs_per_validation,
        transform=train_transforms,
    )
    val_ds = monai.data.CacheDataset(
        cache_rate=1.0,
        data=val_files,
        num_workers=2,
        transform=val_transforms,
    )
    # train_ds = monai.data.Dataset(
    #     data=train_files, transform=train_transforms
    # )
    # val_ds = monai.data.Dataset(
    #     data=val_files, transform=val_transforms
    # )

    train_loader = ThreadDataLoader(
        train_ds, num_workers=12, batch_size=num_images_per_batch, shuffle=True
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1, shuffle=False)

    ckpt = torch.load(args.arch_ckpt)
    node_a = ckpt["node_a"]
    arch_code_a = ckpt["code_a"]
    arch_code_c = ckpt["code_c"]

    dints_space = monai.networks.nets.TopologyInstance(
        channel_mul=1.0,
        num_blocks=12,
        num_depths=4,
        use_downsample=True,
        arch_code=[arch_code_a, arch_code_c],
        device=device,
    )

    model = monai.networks.nets.DiNTS(
        dints_space=dints_space,
        in_channels=input_channels,
        num_classes=output_classes,
        use_downsample=True,
        node_a=node_a,
    )

    model = model.to(device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    post_pred = Compose(
        [EnsureType(), AsDiscrete(argmax=True, to_onehot=output_classes)]
    )
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=output_classes)])

    # loss function
    loss_func = monai.losses.DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        batch=True,
        smooth_nr=0.00001,
        smooth_dr=0.00001,
    )

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate * world_size,
        momentum=0.9,
        weight_decay=0.00004,
    )

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
        )

    if args.checkpoint != None and os.path.isfile(args.checkpoint):
        print(f"[info] fine-tuning pre-trained checkpoint {args.checkpoint:s}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        torch.cuda.empty_cache()
    else:
        print("[info] training from scratch")

    # amp
    if amp:
        from torch.cuda.amp import autocast, GradScaler

        scaler = GradScaler()
        if dist.get_rank() == 0:
            print("[info] amp enabled")

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    idx_iter = 0
    metric_values = list()

    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_root, "Events"))

        with open(os.path.join(args.output_root, "accuracy_history.csv"), "a") as f:
            f.write("epoch\tmetric\tloss\tlr\ttime\titer\n")

    start_time = time.time()
    for epoch in range(num_validation_rounds):
        decay = 0.5 ** np.sum(
            [epoch / num_validation_rounds > learning_rate_milestones]
        )
        lr = learning_rate * decay * world_size
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if dist.get_rank() == 0:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{num_validation_rounds}")
            print(f"learning rate is set to {lr}")

        model.train()
        epoch_loss = 0
        loss_torch = torch.zeros(2, dtype=torch.float, device=device)
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(
                device
            )

            for param in model.parameters():
                param.grad = None

            if amp:
                with autocast():
                    outputs = model(inputs)
                    # if output_classes == 2:
                    #     loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
                    # else:
                    loss = loss_func(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                # if output_classes == 2:
                #     loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
                # else:
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0
            epoch_len = len(train_loader)
            idx_iter += 1

            if dist.get_rank() == 0:
                print(
                    f"[{str(datetime.now())[:19]}] "
                    + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}"
                )
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        # synchronizes all processes and reduce results
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)
        loss_torch = loss_torch.tolist()
        if dist.get_rank() == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            print(
                f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
            )

        if (epoch + 1) % val_interval == 0:
            torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                metric = torch.zeros(
                    (output_classes - 1) * 2, dtype=torch.float, device=device
                )
                metric_sum = 0.0
                metric_count = 0
                metric_mat = []
                val_images = None
                val_labels = None
                val_outputs = None

                _index = 0
                for val_data in val_loader:
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    roi_size = patch_size_valid
                    sw_batch_size = num_sw_batch_size

                    # test time augmentation
                    ct = 1.0
                    with torch.cuda.amp.autocast():
                        pred = sliding_window_inference(
                            val_images,
                            roi_size,
                            sw_batch_size,
                            lambda x: model(x),
                            mode="gaussian",
                            overlap=overlap_ratio,
                        )

                    val_outputs = pred / ct

                    val_outputs = post_pred(val_outputs[0, ...])
                    val_outputs = val_outputs[None, ...]
                    val_labels = post_label(val_labels[0, ...])
                    val_labels = val_labels[None, ...]

                    value = compute_meandice(
                        y_pred=val_outputs, y=val_labels, include_background=False
                    )

                    print(_index + 1, "/", len(val_loader), value)

                    metric_count += len(value)
                    metric_sum += value.sum().item()
                    metric_vals = value.cpu().numpy()
                    if len(metric_mat) == 0:
                        metric_mat = metric_vals
                    else:
                        metric_mat = np.concatenate((metric_mat, metric_vals), axis=0)

                    for _c in range(output_classes - 1):
                        val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                        val1 = 1.0 - torch.isnan(value[0, 0]).float()
                        metric[2 * _c] += val0 * val1
                        metric[2 * _c + 1] += val1

                    _index += 1

                # synchronizes all processes and reduce results
                dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric = metric.tolist()
                if dist.get_rank() == 0:
                    for _c in range(output_classes - 1):
                        print(
                            f"evaluation metric - class {_c + 1:d}:",
                            metric[2 * _c] / metric[2 * _c + 1],
                        )
                    avg_metric = 0
                    for _c in range(output_classes - 1):
                        avg_metric += metric[2 * _c] / metric[2 * _c + 1]
                    avg_metric = avg_metric / float(output_classes - 1)
                    print("avg_metric", avg_metric)

                    if avg_metric > best_metric:
                        best_metric = avg_metric
                        best_metric_epoch = epoch + 1
                        torch.save(
                            model.state_dict(),
                            os.path.join(args.output_root, "best_metric_model.pth"),
                        )
                        print("saved new best metric model")

                        dict_file = {}
                        dict_file["best_avg_dice_score"] = float(best_metric)
                        dict_file["best_avg_dice_score_epoch"] = int(best_metric_epoch)
                        dict_file["best_avg_dice_score_iteration"] = int(idx_iter)
                        with open(
                            os.path.join(args.output_root, "progress.yaml"), "w"
                        ) as out_file:
                            documents = yaml.dump(dict_file, stream=out_file)

                    print(
                        "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, avg_metric, best_metric, best_metric_epoch
                        )
                    )

                    current_time = time.time()
                    elapsed_time = (current_time - start_time) / 60.0
                    with open(
                        os.path.join(args.output_root, "accuracy_history.csv"), "a"
                    ) as f:
                        f.write(
                            "{:d}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.1f}\t{:d}\n".format(
                                epoch + 1,
                                avg_metric,
                                loss_torch_epoch,
                                lr,
                                elapsed_time,
                                idx_iter,
                            )
                        )

                dist.barrier()

            torch.cuda.empty_cache()

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )

    if dist.get_rank() == 0:
        writer.close()

    dist.destroy_process_group()

    return


if __name__ == "__main__":
    main()
