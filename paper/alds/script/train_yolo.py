#!/usr/bin/env python3

# %% import the required packages
import os
import numpy as np
import copy
import torch
import torchvision
from torchvision.datasets.vision import data
import torchprune as tp
from torchprune.util import tensor
from torchprune.util.train import load_checkpoint, save_checkpoint
from torch import multiprocessing as mp
from ultralytics import YOLO
from ultralytics.yolo.v8.detect import DetectionTrainer

# %% initialize the network and wrap it into the NetHandle class




# %% Pre-train the network
# yolo_model.train(data="coco.yaml")

# %% Prune weights on the CPU

def train_pruned_net(rank, m, save_dir):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    m.train(device="0,1,2,3", resume=True, epochs=50, save_dir=save_dir)

if __name__ == "__main__":
    det = DetectionTrainer(
            overrides={"data": "coco.yaml", "model": "yolov8n.pt", "device":0}
    )
    det._setup_train(-1, 1)

    net_name = "yolov8n"
    yolo_model = YOLO("yolov8n.pt")
    net = yolo_model.model
    net = tp.util.net.NetHandle(net, net_name)

    with torch.no_grad():
        device = next(net.parameters()).device
        for in_data in det.test_loader:
            # cache etas with this forward pass
            net(tensor.to(det.preprocess_batch(in_data)["img"], device))
            break

    # %% Setup some stats to track results and retrieve checkpoints
    n_idx = 0  # network index 0
    s_idx = 0  # keep ratio's index
    r_idx = 0  # repetition index
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    # %% Prune filters on the GPU
    print("\n===========================")
    print("Pruning filters with ALDS.")
    net_filter_pruned = tp.ALDSNet(net, det.test_loader, det.criterion)
    net_filter_pruned.cuda()
    for index, keep_ratio in enumerate(np.geomspace(0.9, 0.3, 6).tolist()):
        file_name = f"nets/it{index}.pt"
        if os.path.isfile(file_name):
            load_checkpoint(file_name, net_filter_pruned, loc="cuda:0")
            print(
                f"The loaded network has {net_filter_pruned.size()} parameters and "
                f"{net_filter_pruned.flops()} FLOPs left."
            )
            print("===========================")

            continue
        first_index = index == 0
        net_filter_pruned.compress(keep_ratio=keep_ratio, from_original=first_index)
        net_filter_pruned.compressed_net.register_sparsity_pattern()
        net_filter_pruned.cpu()
        print(
            f"The network has {net_filter_pruned.size()} parameters and "
            f"{net_filter_pruned.flops()} FLOPs left."
        )
        print("===========================")

        # %% Retrain the filter-pruned network now.

        # Retrain the filter-pruned network now on the GPU
        net_filter_pruned = net_filter_pruned.cuda()
        yolo_model.model = net_filter_pruned.compressed_net.torchnet
        mp.spawn(train_pruned_net, nprocs=4, args=(yolo_model, f"it{index}"))
        yolo_model.to("cuda:0")
        net_filter_pruned.compressed_net.torchnet = yolo_model.model
        save_checkpoint(f"nets/it{index}.pt", net_filter_pruned, 50)

    # %% Test at the end
    # print("\nTesting on test data set:")
    # loss, acc1, acc5 = trainer.test(net_filter_pruned)
    # print(f"Loss: {loss:.4f}, Top-1 Acc: {acc1*100:.2f}%, Top-5: {acc5*100:.2f}%")

    # # Put back to CPU
