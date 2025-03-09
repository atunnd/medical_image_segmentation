import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR

import torch 
from torch import nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.cuda import amp
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms import RandAugment
from torch.utils.tensorboard import SummaryWriter

from test_util import test_all_case



from networks.vnet import VNet
from networks.ema import ModelEMA
from utils.losses import dice_loss
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler, RandomCrop_MPL, RandomRotFlip_MPL, ToTensor_MPL, RandomNoise_MPL
from utils.util import AverageMeter, save_checkpoint, accuracy, model_load_state_dict

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='data path')
parser.add_argument('--name', type=str, default='LA', help='Name of experiment')
parser.add_argument('--labeled_sample', type=int, default=3, help='Number of labeled samples')
parser.add_argument('--exp', type=str,  default='vnet_LA', help='model_name')
parser.add_argument('--total_steps', type=int,  default=4, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=3, help='batch_size per gpu')
parser.add_argument('--eval-step', default=4, type=int, help='number of eval steps to run')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--dataset', default='LA', type=str, choices=['LA'], help='dataset name')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes to load parallel data')
parser.add_argument("--save_path", type=str, default='model')
parser.add_argument('--evaluate', action='store_true', help='evaluating on testset')
parser.add_argument('--finetune', action='store_true', help='finetune model on labeled dataset')

# training MPL
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')
parser.add_argument('--grad-clip', default=1e9, type=float, help='gradient norm clipping')

parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
parser.add_argument('--teacher_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")

# finetune after training
parser.add_argument('--finetune-epochs', default=3, type=int, help='finetune epochs')
parser.add_argument('--finetune-batch-size', default=3, type=int, help='finetune batch size')
parser.add_argument('--finetune-lr', default=3e-5, type=float, help='finetune learning late')
parser.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
parser.add_argument('--finetune-momentum', default=0.9, type=float, help='finetune SGD Momentum')


args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.total_steps
base_lr = args.base_lr

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = ("cpu" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def train_loop(args, labeled_loader, unlabeled_loader, test_loader, finetune_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler):
    
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(labeled_loader)))

    iter_num = 0
    max_epoch = args.total_steps//len(labeled_loader)+1
    lr_ = base_lr

    

    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.labeled_sample}")
    logger.info(f"   Total steps = {args.total_steps}")        

    for epoch_num in tqdm(range(max_epoch), ncols=70):

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        time1 = time.time()
        teacher_model.zero_grad()
        student_model.zero_grad()

        if epoch_num % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()

        l_sample = next(labeled_iter)
        ul_sample = next(unlabeled_iter)

        images_l = l_sample['image'].to(device)
        targets = l_sample['label'].to(device)
        images_uw = ul_sample['weak_aug'].to(device)
        images_us = ul_sample['strong_aug'].to(device)

        with torch.amp.autocast(device_type, enabled=args.amp):
            batch_size = images_l.shape[0]
            t_images = torch.cat((images_l, images_uw, images_us))
            t_logits = teacher_model(t_images)
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
            del t_logits

            t_loss_l = criterion(t_logits_l, targets)

            soft_pseudo_labels = torch.softmax(t_logits_uw.detach() / args.temperature, dim=1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_labels, dim=1)
            mask = max_probs.ge(args.threshold).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_labels * torch.log_softmax(t_logits_us,dim=1)).sum(dim=1)*mask
            )
            
            weight_u = args.lambda_u * min(1., (epoch_num + 1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u + t_loss_u

            s_images = torch.cat((images_l, images_us))
            s_logits = student_model(s_images)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits

            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
            s_loss = criterion(s_logits_us, hard_pseudo_label)


        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        
        s_scaler.step(t_optimizer)
        s_scaler.update()
        s_scheduler.step()
        

        if args.ema > 0:
            avg_student_model.update_parameters(student_model)
        
        with torch.amp.autocast(device_type, enabled=args.amp):
            with torch.no_grad():
                s_logits_l = student_model(images_l)
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)

            dot_product = s_loss_l_new - s_loss_l_old

            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=1)
            t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
            t_loss = t_loss_uda + t_loss_mpl
        
        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()
        

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())

        pbar.set_description(
            f"Train Iter: {epoch_num+1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()

        num_eval = epoch_num // args.eval_step   
        if (epoch_num + 1) % args.eval_step == 0:
            pbar.close()
            writer.add_scalar("train/1.s_loss", s_losses.avg, num_eval)
            writer.add_scalar("train/2.t_loss", t_losses.avg, num_eval)
            writer.add_scalar("train/3.t_labeled", t_losses_l.avg, num_eval)
            writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, num_eval)
            writer.add_scalar("train/5.t_mpl", t_losses_mpl.avg, num_eval)
            writer.add_scalar("train/6.mask", mean_mask.avg, num_eval)

            test_model = avg_student_model if avg_student_model is not None else student_model
            test_loss = evaluate(args, test_loader, test_model, criterion)

            writer.add_scalar("test/loss", test_loss, num_eval)

            save_checkpoint(args, {
                    'step': epoch_num + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
            })

    del t_scaler, t_scheduler, t_optimizer, teacher_model, labeled_loader, unlabeled_loader
    del s_scaler, s_scheduler, s_optimizer
    ckpt_name = f'{args.save_path}/{args.name}_last.pth.tar'
    loc = f'cuda:{args.gpu}'
    checkpoint = torch.load(ckpt_name, map_location=device)
    logger.info(f"=> loading checkpoint '{ckpt_name}'")
    if checkpoint['avg_state_dict'] is not None:
        model_load_state_dict(student_model, checkpoint['avg_state_dict'])
    else:
        model_load_state_dict(student_model, checkpoint['student_state_dict'])
    finetune(args, writer, finetune_loader, test_loader, student_model, criterion)
    return

def criterion_function(pred, targets):
        loss_seg = F.cross_entropy(pred, targets)
        outputs_soft = F.softmax(pred, dim=1)

        loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], targets == 1)
        t_loss_l = 0.5*(loss_seg + loss_seg_dice)

        return t_loss_l
    
def evaluate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    test_iter = tqdm(test_loader)
    with torch.no_grad():
        end = time.time()
        for step, sample in enumerate(test_iter):
            data_time.update(time.time() - end)
            images, targets = sample['image'], sample['label']
            print("test image: ", images.shape)
            batch_size = images.shape[0]
            images = images.to(device)
            targets = targets.to(device)
            with torch.amp.autocast(device_type, enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                    f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                    f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}.")

    test_iter.close()
    return losses.avg
    
def finetune(args, writer,  finetune_loader, test_loader, model, criterion):
        model.drop = nn.Identity()
        train_sampler = RandomSampler 
        labeled_loader = finetune_loader
        optimizer = optim.SGD(model.parameters(),
                            lr=args.finetune_lr,
                            momentum=args.finetune_momentum,
                            weight_decay=args.finetune_weight_decay,
                            nesterov=True)
        scaler = torch.amp.GradScaler(device, enabled=args.amp)

        logger.info("***** Running Finetuning *****")
        logger.info(f"   Finetuning steps = {len(labeled_loader)*args.finetune_epochs}")

        for epoch in range(args.finetune_epochs):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            model.train()
            end = time.time()
            labeled_iter = tqdm(labeled_loader)
            for step, sample in enumerate(labeled_iter):
                data_time.update(time.time() - end)
                images, targets = sample['image'], sample['label']
                print("\nfinetune images: ", images.shape)
                print("targets: ", targets.shape)
                batch_size = images.shape[0]
                images = images.to(device)
                targets = targets.to(device)
                with torch.amp.autocast(device_type, enabled=args.amp):
                    model.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                losses.update(loss.item(), batch_size)
                batch_time.update(time.time() - end)
                labeled_iter.set_description(
                    f"Finetune Epoch: {epoch+1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                    f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
            labeled_iter.close()

            writer.add_scalar("finetune/train_loss", losses.avg, epoch)
            test_loss = evaluate(args, test_loader, model, criterion)
            writer.add_scalar("finetune/test_loss", test_loss, epoch)

            save_checkpoint(args, {
                    'step': step + 1,
                    'student_state_dict': model.state_dict(),
                    'avg_state_dict': None,
                    'student_optimizer': optimizer.state_dict(),
                },  finetune=True)
        return
        


def main():
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       num=args.labeled_sample,
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    
    db_unlabeled_train = LAHeart(base_dir=train_data_path,
                                    split='train',
                                    unlabeled_num=5,   #80-args.labeled_sample,
                                    mpl_mode=True,
                                    transform= transforms.Compose([
                                        RandomRotFlip_MPL(),
                                        RandomCrop_MPL(patch_size),
                                        RandomNoise_MPL(),
                                        ToTensor_MPL()
                                    ]))
    db_test = LAHeart(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    db_finetune = LAHeart(base_dir=train_data_path,
                                    split='train',
                                    num=args.labeled_sample,
                                    transform = transforms.Compose([
                                    RandomRotFlip(),
                                    RandomCrop(patch_size),
                                    ToTensor(),
                                    ]))

    teacher_model = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    student_model = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)


    no_decay = ['bn']
    teacher_parameters = [
        {'params': [p for n, p in teacher_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in teacher_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    student_parameters = [
        {'params': [p for n, p in student_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in student_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    avg_student_model = None
    if args.ema > 0:
        avg_student_model = ModelEMA(student_model, args.ema)

    t_optimizer = optim.SGD(teacher_parameters,
                            lr=args.teacher_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)
    s_optimizer = optim.SGD(student_parameters,
                            lr=args.student_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)
    
    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)
    t_scaler = torch.amp.GradScaler(enabled=args.amp)
    s_scaler = torch.amp.GradScaler(enabled=args.amp)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    unlabled_trainloader = DataLoader(db_unlabeled_train, batch_size=batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False,  num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    finetune_loader =  DataLoader(db_finetune, batch_size=batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
   
    criterion = criterion_function

    if args.evaluate:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, trainloader, unlabled_trainloader
        del s_scaler, s_scheduler, s_optimizer
        
        ckpt_name = f'{args.save_path}/{args.name}_last.pth.tar'
        loc = f'cuda:{args.gpu}'
        checkpoint = torch.load(ckpt_name, map_location=device)
        logger.info(f"=> loading checkpoint '{ckpt_name}'")
        if checkpoint['avg_state_dict'] is not None:
            model_load_state_dict(student_model, checkpoint['avg_state_dict'])
        else:
            model_load_state_dict(student_model, checkpoint['student_state_dict'])

        with open(args.root_path + '/../test.list', 'r') as f:
            image_list = f.readlines()
        image_list = [args.root_path +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]

        avg_metric = test_all_case(student_model, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=False, test_save_path=None)

        # evaluate(args, testloader, student_model, criterion)

        return

    train_loop(args, trainloader, unlabled_trainloader,  testloader, finetune_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler)

    return
    
if __name__ == "__main__":
    main()      
    
        


    