import copy
import math
import random
from os import environ
from platform import system

import cv2
import numpy
import torch
from PIL import Image
from PIL import ImageEnhance, ImageOps

max_value = 10.0


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def plot_lr(args, optimizer, scheduler):
    import copy
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        y.append(optimizer.param_groups[0]['lr'])
        scheduler.step(epoch + 1, optimizer)

    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('epoch')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr2.png', dpi=200)
    pyplot.close()


def set_params(model):
    p1 = []
    p2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            p1.append(param)
        else:
            p2.append(param)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': 5E-5}]


@torch.no_grad()
def accuracy(outputs, targets, top_k):
    results = []
    outputs = outputs.topk(max(top_k), 1, True, True)[1].t()
    outputs = outputs.eq(targets.view(1, -1).expand_as(outputs))

    for k in top_k:
        correct = outputs[:k].reshape(-1)
        correct = correct.float().sum(0, keepdim=True)
        results.append(correct.mul_(100.0 / targets.size(0)))
    return results


def resample():
    return random.choice((Image.Resampling.BICUBIC, Image.Resampling.BILINEAR))


def equalize(image, _):
    return ImageOps.equalize(image)


def invert(image, _):
    return ImageOps.invert(image)


def normalize(image, _):
    return ImageOps.autocontrast(image)


def rotate(image, m):
    m = (m / max_value) * 30.0

    if random.random() > 0.5:
        m *= -1

    return image.rotate(m, resample=resample())


def shear_x(image, m):
    m = (m / max_value) * 0.30

    if random.random() > 0.5:
        m *= -1

    return image.transform(image.size, Image.Transform.AFFINE, (1, m, 0, 0, 1, 0), resample=resample())


def shear_y(image, m):
    m = (m / max_value) * 0.30

    if random.random() > 0.5:
        m *= -1

    return image.transform(image.size, Image.Transform.AFFINE, (1, 0, 0, m, 1, 0), resample=resample())


def translate_x(image, m):
    m = (m / max_value) * 0.45

    if random.random() > 0.5:
        m *= -1

    pixels = m * image.size[0]
    return image.transform(image.size, Image.Transform.AFFINE, (1, 0, pixels, 0, 1, 0), resample=resample())


def translate_y(image, m):
    m = (m / max_value) * 0.45

    if random.random() > 0.5:
        m *= -1

    pixels = m * image.size[1]
    return image.transform(image.size, Image.Transform.AFFINE, (1, 0, 0, 0, 1, pixels), resample=resample())


def brightness(image, m):
    m = (m / max_value) * 1.8 + 0.1
    return ImageEnhance.Brightness(image).enhance(m)


def color(image, m):
    m = (m / max_value) * 1.8 + 0.1
    return ImageEnhance.Color(image).enhance(m)


def contrast(image, m):
    m = (m / max_value) * 1.8 + 0.1
    return ImageEnhance.Contrast(image).enhance(m)


def poster(image, m):
    m = int((m / max_value) * 4)
    if m >= 8:
        return image
    return ImageOps.posterize(image, m)


def sharpness(image, m):
    m = (m / max_value) * 1.8 + 0.1
    return ImageEnhance.Sharpness(image).enhance(m)


def solar(image, m):
    m = min(256, int((m / max_value) * 256))
    return ImageOps.solarize(image, m)


def solar_add(image, m):
    lut = []
    m = min(128, int((m / max_value) * 110))
    for i in range(256):
        if i < 128:
            lut.append(min(255, i + m))
        else:
            lut.append(i)

    if image.mode in ("L", "RGB"):
        if image.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)

    return image


class Resize:
    def __init__(self, input_size):
        self.size = input_size
        self.scale = (0.08, 1.0)
        self.ratio = (3. / 4., 4. / 3.)

    def __call__(self, image):
        size = self.size
        i, j, h, w = self.random_size(image.size)
        image = image.crop(box=(j, i, j + w, i + h))
        return image.resize([size, size], resample())

    def random_size(self, size):
        scale = self.scale
        ratio = self.ratio
        for _ in range(10):
            target_area = random.uniform(*scale) * size[0] * size[1]
            aspect_ratio = math.exp(random.uniform(*(math.log(ratio[0]), math.log(ratio[1]))))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        if (size[0] / size[1]) < min(ratio):
            w = size[0]
            h = int(round(w / min(ratio)))
        elif (size[0] / size[1]) > max(ratio):
            h = size[1]
            w = int(round(h * max(ratio)))
        else:
            w = size[0]
            h = size[1]
        i = (size[1] - h) // 2
        j = (size[0] - w) // 2
        return i, j, h, w


class RandomAugment:
    def __init__(self, mean=9.0, sigma=0.5, n=2, p=0.5):
        self.p = p
        self.n = n
        self.mean = mean
        self.sigma = sigma
        self.transform = (equalize, invert, normalize,
                          rotate, shear_x, shear_y, translate_x, translate_y,
                          brightness, color, contrast, poster, sharpness, solar, solar_add)

    def __call__(self, image):
        if random.random() > self.p:
            return image
        for transform in numpy.random.choice(self.transform, self.n):
            m = numpy.random.normal(self.mean, self.sigma)
            m = min(max_value, max(0.0, m))

            image = transform(image, m)
        return image


class RandomErase:
    def __init__(self, p=0.2):
        self.p = p
        self.min_area = 0.02
        self.max_area = 1 / 3
        self.aspect_ratio = (math.log(0.3), math.log(1 / 0.3))

    def __call__(self, image):
        if random.random() > self.p:
            return image

        size = image.size()
        area = size[1] * size[2]

        for _ in range(10):
            target_area = random.uniform(self.min_area, self.max_area) * area
            aspect_ratio = math.exp(random.uniform(*self.aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if h < size[1] and w < size[2]:
                y = random.randint(0, size[1] - h)
                x = random.randint(0, size[2] - w)
                image[:, y:y + h, x:x + w] = torch.empty((size[0], h, w),
                                                         dtype=image.dtype,
                                                         device=image.device).normal_()
                break
        return image


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        self.num = self.num + n
        self.sum = self.sum + v * n
        self.avg = self.sum / self.num


class EMA:
    """
    [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage]
    """

    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.model = copy.deepcopy(model).eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module

        m_std = model.state_dict().values()
        e_std = self.model.state_dict().values()

        for m, e in zip(m_std, e_std):
            e.copy_(self.decay * e + (1. - self.decay) * m)


class StepLR:
    def __init__(self, lr):
        self.lr = lr
        self.decay_rate = 0.98
        self.decay_epochs = 2.4
        self.warmup_epochs = 3.0
        self.warmup_lr_init = 1E-6

    def step(self, epoch, optimizer):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr_init + epoch * (self.lr - self.warmup_lr_init) / self.warmup_epochs
        else:
            lr = self.lr * (self.decay_rate ** ((epoch - self.warmup_epochs) // self.decay_epochs))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class RMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=1E-6, alpha=0.9, eps=1E-3, weight_decay=0.0,
                 momentum=0.9, centered=False, decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        centered=centered, decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for param_group in self.param_groups:
            param_group.setdefault('momentum', 0)
            param_group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(param.data)
                    if param_group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                    if param_group['centered']:
                        state['grad_avg'] = torch.zeros_like(param.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - param_group['alpha']

                state['step'] += 1

                if param_group['weight_decay'] != 0:
                    if 'decoupled_decay' in param_group and param_group['decoupled_decay']:
                        param.data.add_(param.data, alpha=-param_group['weight_decay'])
                    else:
                        grad = grad.add(param.data, alpha=param_group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if param_group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(param_group['eps']).sqrt_()
                else:
                    avg = square_avg.add(param_group['eps']).sqrt_()

                if param_group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in param_group and param_group['lr_in_momentum']:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg, value=param_group['lr'])
                        param.data.add_(-buf)
                    else:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
                        param.data.add_(-param_group['lr'], buf)
                else:
                    param.data.addcdiv_(grad, avg, value=-param_group['lr'])

        return loss


class CrossEntropyLoss(torch.nn.Module):
    """
    NLL Loss with label smoothing.
    """

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, outputs, targets):
        prob = self.softmax(outputs)
        mean = torch.mean(prob, dim=-1)

        index = torch.unsqueeze(targets, dim=1)

        nll_loss = torch.gather(prob, -1, index)
        nll_loss = torch.squeeze(nll_loss, dim=1)

        return ((self.epsilon - 1) * nll_loss - self.epsilon * mean).mean()
