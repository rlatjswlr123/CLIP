import torch
import numpy as np
import random
import argparse
import datetime


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.requires_grad:
            p.grad.data = p.grad.data.float()


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100, help='Epoch 수')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--train_path', type=str, 
                        # default='/data4/issue_images_split/all_train')
                        # default='/data4/issue_images_split/all_train')
                        default="/data4/issue_images_package_split/train")
                        # default="/data4/issue_images_split_special_harzard/issue_images/train")
                        # default="/data4//issue_images_package/train")
    parser.add_argument('--test_path', type=str, 
                        # default='/data4/issue_images_split/val')
                        # default='/data4/issue_images_split/val')
                        default="/data4/issue_images_package_split/val")
                        # default="/data4/issue_images_split_special_harzard/issue_images/train")
    parser.add_argument('--val_interval', type=int, default=1,
                        help='몇 epoch 마다 validation 수치를 뽑을지')
    parser.add_argument('--print_log_by_iter', type=int, default=50)
    parser.add_argument('--save_log_dir', type=str, 
                        default=f"test_{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='ViT-L/14',
                        help='ViT-B/32, ViT-B/16, ViT-L/14, RN50 입력 가능')
    parser.add_argument('--model_save_mode', type=str, default='loss',
                        help='precision, recall, acc 입력가능')
    parser.add_argument('--is_tsne', type=bool, default=True,
                        help='tsne를 통해 image embedding을 visualization 할지')
    parser.add_argument('--tsne_interval', type=int, default=5,
                        help='몇 epoch 마다 tsne를 뽑을지')
        
    return parser


def get_evaluation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--test_path', type=str, 
                        default='/home/sjkim/test')
    parser.add_argument('--save_log_dir', type=str, 
                        default=f"./runs/clip_evaluation_{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}")
    parser.add_argument('--model_name', type=str, default='ViT-B/32',
                        help='ViT-B/32, ViT-B/16, ViT-L/14, RN50 입력 가능')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='load할 weight 경로')

    return parser