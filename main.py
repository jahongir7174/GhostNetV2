import csv
import os
from argparse import ArgumentParser
from warnings import filterwarnings

import torch
import tqdm
from timm import utils
from torch.utils import data
from torchvision import transforms

from nets import nn
from utils import util
from utils.dataset import Dataset

filterwarnings("ignore")

data_dir = os.path.join('..', 'Dataset', 'IMAGENET')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def lr(args):
    return 0.256 * args.batch_size * args.world_size / 4096


def train(args):
    # Model
    model = nn.GhostNetV2().cuda()
    ema_m = util.EMA(model)

    amp_scale = torch.cuda.amp.GradScaler()
    optimizer = util.RMSprop(util.set_params(model))

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank])

    scheduler = util.StepLR(lr(args))
    criterion = util.CrossEntropyLoss().cuda()

    sampler = None
    dataset = Dataset(os.path.join(data_dir, 'train'),
                      transforms.Compose([util.Resize(input_size=args.input_size),
                                          util.RandomAugment(mean=9.0, sigma=0.5),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(), normalize,
                                          util.RandomErase()]))

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    best = 0
    loader = data.DataLoader(dataset, args.batch_size, not args.distributed,
                             sampler=sampler, num_workers=8, pin_memory=True)

    with open('weights/step.csv', 'w') as log:

        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'acc@1', 'acc@5',
                                                     'train_loss', 'val_loss'])
            logger.writeheader()

        for epoch in range(args.epochs):
            if args.distributed:
                sampler.set_epoch(epoch)

            p_bar = loader
            avg_loss = util.AverageMeter()

            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                p_bar = tqdm.tqdm(loader, total=len(loader))

            model.train()
            for samples, targets in p_bar:
                samples = samples.cuda()
                targets = targets.cuda()

                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                    loss = criterion(outputs, targets)

                optimizer.zero_grad()

                amp_scale.scale(loss).backward()
                amp_scale.step(optimizer)
                amp_scale.update()

                torch.cuda.synchronize()
                ema_m.update(model)

                if args.distributed:
                    loss = utils.reduce_tensor(loss.data, args.world_size)

                avg_loss.update(loss.item(), samples.size(0))

                if args.local_rank == 0:
                    gpus = '%.4gG' % (torch.cuda.memory_reserved() / 1E9)
                    desc = ('%10s' * 2 + '%10.3g') % ('%g/%g' % (epoch + 1, args.epochs), gpus, avg_loss.avg)
                    p_bar.set_description(desc)

            scheduler.step(epoch + 1, optimizer)

            if args.local_rank == 0:
                last = test(args, ema_m.model)
                logger.writerow({'acc@1': str(f'{last[1]:.3f}'),
                                 'acc@5': str(f'{last[2]:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3),
                                 'val_loss': str(f'{last[0]:.3f}'),
                                 'train_loss': str(f'{avg_loss.avg:.3f}')})
                log.flush()

                # Update best Acc
                if best < last[1]:
                    best = last[1]

                # Save last, best and delete
                save = {'model': ema_m.model}
                torch.save(save, f='weights/last.pt')
                if best == last[1]:
                    torch.save(save, f='weights/best.pt')
                del save

    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, model=None):
    if model is None:
        model = torch.load('weights/best.pt')
        model = model['model'].float().fuse()
        model.cuda()

    model.eval()

    criterion = torch.nn.CrossEntropyLoss().cuda()

    dataset = Dataset(os.path.join(data_dir, 'val'),
                      transforms.Compose([transforms.Resize(args.input_size + 32),
                                          transforms.CenterCrop(args.input_size),
                                          transforms.ToTensor(), normalize]))

    loader = data.DataLoader(dataset, batch_size=32, num_workers=8, pin_memory=True)

    top1 = util.AverageMeter()
    top5 = util.AverageMeter()
    avg_loss = util.AverageMeter()

    for samples, targets in tqdm.tqdm(loader, ('%10s' * 3) % ('acc@1', 'acc@5', 'loss')):
        samples = samples.cuda()
        targets = targets.cuda()

        with torch.cuda.amp.autocast():
            outputs = model(samples)

        acc1, acc5 = util.accuracy(outputs, targets, top_k=(1, 5))
        torch.cuda.synchronize()

        top1.update(acc1.item(), samples.size(0))
        top5.update(acc5.item(), samples.size(0))
        avg_loss.update(criterion(outputs, targets).item(), samples.size(0))

    acc1, acc5 = top1.avg, top5.avg
    print('%10.4g' * 3 % (acc1, acc5, avg_loss.avg))
    if model is None:
        torch.cuda.empty_cache()
    else:
        return avg_loss.avg, acc1, acc5


def profile(args):
    import thop
    model = nn.GhostNetV2().fuse()
    shape = (1, 3, args.input_size, args.input_size)

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model=model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[flops, num_params], format="%.3f")
    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')
    if args.benchmark:
        # Latency
        model = nn.GhostNetV2().fuse()
        model.eval()

        x = torch.zeros(shape)
        for i in range(10):
            model(x)
        total = 0
        import time
        for i in range(1_000):
            start = time.perf_counter()
            with torch.no_grad():
                model(x)
            total += time.perf_counter() - start

        print(f"Latency: {total / 1_000 * 1_000:.3f} ms")


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--epochs', default=450, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    profile(args)

    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == '__main__':
    main()
