# %%
from functools import wraps
import json
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
from contextlib import contextmanager

# print('GPU USE IS TURNED OFF! (in utilities.py)')
# if False:
if torch.cuda.is_available():
    # Device configuration
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    # Set default dtype to float32
    torch.set_default_dtype(torch.float)


def set_device(newdevice):
    global device
    device = newdevice
    if 'cuda' in device:
        torch.cuda.set_device(device)


printing_enabled = True
if not printing_enabled:
    print("printing is DISABLED in utilities.py")

logging.basicConfig(level=logging.INFO)


with open('config.json') as config:
    config_data = json.load(config)
os.environ["DATASETS"] = config_data["dataset_path"]

# %%


def getRandomImagePaths(seed=None, key=None, i=None, set='train'):
    from flow_datasets import getSintelTrain, getSintelTest
    np.random.seed(seed)
    dataset = getSintelTrain(
        'final') if set == 'train' else getSintelTest('final')
    key = key or np.random.choice(list(dataset.keys()))
    i = i or np.random.randint(0, len(dataset[key]["images"])-1)
    # if test only return images
    if set == 'test':
        return dataset[key]["images"][i], dataset[key]["images"][i+1]
    print(f"scene={key}, i={i}")
    return dataset[key]["images"][i], dataset[key]["images"][i+1], dataset[key]["flows"][i]


def read_gray_image(path, type='tensor'):
    if type.lower() == 'tensor' or type.lower() == 'torch':
        from torchvision.io import read_image, ImageReadMode
        return (read_image(path, ImageReadMode.GRAY)/255).to(device).squeeze()
    elif type.lower() == 'array' or type.lower() == 'numpy':
        from matplotlib.image import imread
        return imread(path).mean(-1)
    else:
        raise ValueError(
            f"{type} is not a valid type. Use 'tensor' or 'numpy'")
# %%


def progress_bar(amount, of, title='Percent', msg='', length=25):
    if printing_enabled:
        from sys import stdout
        """
        creates a progressbar. No prints during the progress allowed else it looks bad.s
        """
        percent = amount/of
        block = int(round(length*percent))
        text = f"\r{title}: |{'█'*block+'-'*(length-block)}| {round(100*percent*100)/100:<04}% {msg}"
        stdout.write(text)
        stdout.flush()
        if percent >= 1:
            print(' Done!\n')


def clip(x, down=0, up=1):
    """clipping that doesn't interrupt gradients using relu functions

    Arguments:
        x {Any} -- input

    Keyword Arguments:
        down {float} -- lower bound for clip (default: {0})
        up {float} -- upper bound for clip (default: {1})

    Returns:
        x -- clipped x
    """
    x = torch.relu(x+down)+down
    x = -torch.relu(-x+up)+up
    # assert (x >= down).all() and (x <= up).all(), "clipping didn't work"
    return x


def batch_list(l: list, batch_size: int, images=False) -> list:
    """batches given list in multiple lists. The last list will contain the rest of the elements that cant fit in packages of `batch_size`

    Arguments:
        l {list} -- 
        batch_size {int} -- 
        images {bool} -- If images=True, the last image of the previous batch is included at the start of the next batch

    Returns:
        list -- batched list
    """
    if images:
        res = []
        for i in range(len(l)//batch_size):
            res.append(l[i*batch_size:(i+1)*batch_size+1])

        i = len(l)//batch_size-1
        if len(l) > (i+1)*batch_size+1:
            res.append(l[(i+1)*batch_size:])
        return res
    return [l[i*batch_size:(i+1)*batch_size] for i in range(len(l)//batch_size+1) if i*batch_size < len(l)]


def itercount(iterator, condition):
    """counts the number of elements in iterator where condition is true

    Arguments:
        iterator {iterator}
        condition {function}

    Returns:
        int -- count of elements
    """
    count = 0
    for x in iterator:
        if condition(x):
            count += 1
    return count


def isNumber(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def getIndexList(list, indices):
    return [x for i, x in enumerate(list) if i in indices]


def flowFromUV(u, v):
    if isinstance(u, torch.Tensor):
        return torch.cat((u.unsqueeze(-1), v.unsqueeze(-1)), dim=-1)
    elif isinstance(u, np.ndarray):
        return np.concatenate((np.expand_dims(u, -1), np.expand_dims(v, -1)), axis=-1)
    else:
        raise ValueError('Not torch tensor or numpy array')


def convertTypes(to):
    assert not callable(
        to), f'Need to pass a type to convert to instead of function {to.__name__}'
    _torch = 'torch' in to or 'tensor' in to
    _numpy = 'numpy' in to or 'np' in to

    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            newargs = []
            newdict = {}

            for arg in args:
                if _torch and isinstance(arg, np.ndarray):
                    newargs.append(torch.from_numpy(arg).float().to(device))
                elif _numpy and torch.is_tensor(arg):
                    newargs.append(arg.detach().cpu().numpy())
                else:
                    newargs.append(arg)

            for key, item in kwargs.items():
                if _torch and isinstance(item, np.ndarray):
                    newitem = torch.from_numpy(item).float().to(device)
                elif _numpy and torch.is_tensor(item):
                    newitem = item.detach().cpu().numpy()
                else:
                    newitem = item
                newdict[key] = newitem

            return fn(*newargs, **newdict)
        return wrapper
    return decorate


# %% image derivatives
@convertTypes('tensor')
def get_f_x(img, asnumpy=False):
    """ passes x sobel operator over the image """
    # fallunterscheidung to not mess up batch size
    if len(img.shape) == 2:
        logging.debug(f'unsqueezing because len(img.shape)={len(img.shape)}')
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        logging.debug(f'unsqueezing because len(img.shape)={len(img.shape)}')
        img = img.unsqueeze(1)

    _, inchannel, _, _ = img.size()
    sobelx = -1/8*torch.tensor(
        [[1., 2, 1],
         [0, 0, 0],
         [-1, -2, -1]]*inchannel, device=img.device
    ).transpose(-1, -2)
    sobelx = sobelx.reshape((1, inchannel, 3, 3))
    logging.debug(f'{img.size()}')
    logging.debug(f'{sobelx.size()}')
    img_x = torch.nn.functional.conv2d(img, sobelx, padding='same')
    if asnumpy:
        return img_x.cpu().detach().numpy()
    return img_x


@torch.jit.script
def jit_fx(img):
    """ passes x sobel operator over the image """
    # fallunterscheidung to not mess up batch size
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(1)

    _, inchannel, _, _ = img.size()
    sobelx = -1/8*torch.tensor(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]]*inchannel, device=img.device
    ).transpose(-1, -2).float()
    sobelx = sobelx.reshape((1, inchannel, 3, 3))
    img_x = torch.nn.functional.conv2d(img, sobelx, padding='same')
    return img_x


@convertTypes('tensor')
def get_f_y(img, asnumpy=False):
    """ passes y sobel operator over the image """
    if len(img.shape) == 2:
        logging.debug(f'unsqueezing because len(img.shape)={len(img.shape)}')
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        logging.debug(f'unsqueezing because len(img.shape)={len(img.shape)}')
        img = img.unsqueeze(1)

    batch_size, inchannel, _, _ = img.size()
    sobely = -1/8*torch.tensor(
        [[1., 2, 1],
         [0, 0, 0],
         [-1, -2, -1]]*inchannel, device=img.device
    )
    sobely = sobely.reshape((1, inchannel, 3, 3))
    logging.debug(f'{img.size()}')
    logging.debug(f'{sobely.size()}')
    img_y = torch.nn.functional.conv2d(img, sobely, padding='same')
    if asnumpy:
        return img_y.cpu().detach().numpy()
    return img_y


@torch.jit.script
def jit_fy(img):
    """ passes y sobel operator over the image """
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(1)

    _, inchannel, _, _ = img.size()
    sobely = -1/8*torch.tensor(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]]*inchannel, device=img.device
    ).float()
    sobely = sobely.reshape((1, inchannel, 3, 3))
    img_y = torch.nn.functional.conv2d(img, sobely, padding='same')
    return img_y


@convertTypes('tensor')
def get_lapl_f(img, asnumpy=False):
    """ passes laplace sobel operator over the image """
    if len(img.shape) == 2:
        logging.debug(f'unsqueezing because len(img.shape)={len(img.shape)}')
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        logging.debug(f'unsqueezing because len(img.shape)={len(img.shape)}')
        img = img.unsqueeze(1)

    batch_size, inchannel, _, _ = img.size()
    lapl = -1/6*torch.tensor(
        [[1., 4, 1],
         [4, -20, 4],
         [1, 4, 1]]*inchannel, device=img.device
    )
    lapl = lapl.reshape((1, inchannel, 3, 3))
    logging.debug(f'{img.size()}')
    logging.debug(f'{lapl.size()}')
    img_y = torch.nn.functional.conv2d(img, lapl, padding='same')
    if asnumpy:
        return img_y.cpu().detach().numpy()
    return img_y


@torch.jit.script
def get_lapl_f(img):
    """ passes laplace sobel operator over the image """
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(1)

    _, inchannel, _, _ = img.size()
    lapl = -1/6*torch.tensor(
        [[1, 4, 1],
         [4, -20, 4],
         [1, 4, 1]]*inchannel, device=img.device
    ).float()
    lapl = lapl.reshape((1, inchannel, 3, 3))
    img_y = torch.nn.functional.conv2d(img, lapl, padding='same')
    return img_y


def get_f_z(img1, img2):
    """ 
    returns (img2-img1)/2
    """
    return (img2-img1)/2.


@torch.jit.script
def jit_fz(img1, img2):
    """ 
    returns (img2-img1)/2
    """
    return (img2-img1)/2.


@convertTypes('tensor')
def avg_EndPointError(flow, flow_orig, relative=False):
    """
    returns the average endpoint error of the flow.
    if relative is True, the error is relative to the second argument.
    """
    if len(flow.shape) == 3:  # if not batched
        if relative:
            return torch.sqrt(((flow-flow_orig)**2).sum(dim=-1)).mean(dim=(0, 1))/torch.sqrt((flow_orig**2).sum(dim=-1)).mean(dim=(0, 1))
        return torch.sqrt(((flow-flow_orig)**2).sum(dim=-1)).mean(dim=(0, 1))
    if len(flow.shape) == 4:  # if batched
        if relative:
            return torch.sqrt(((flow-flow_orig)**2).sum(dim=-1)).mean(dim=(1, 2))/torch.sqrt((flow_orig**2).sum(dim=-1)).mean(dim=(1, 2))
        return torch.sqrt(((flow-flow_orig)**2).sum(dim=-1)).mean(dim=(1, 2))
    raise ValueError("flow doesn't have flow shape")


def make_gif(frame_folder):
    """Generates gif from all images in "frame_folder"

    Args:
        frame_folder (string)
    """
    import os
    import imageio
    from PIL import Image
    import logging

    # build gif
    with imageio.get_writer('progress.gif', mode='I') as writer:
        names = os.listdir(f"{frame_folder}")
        names.sort()
        length = len(names)
        for i, filename in enumerate(names):
            progress_bar(i, length)
            logging.debug(frame_folder+'/'+filename)
            image = imageio.imread(frame_folder+'/'+filename)
            writer.append_data(image)


@convertTypes('numpy')
def show_images(*imgs: tuple, names=None, colorbars=False, wait=False, show=False, save=False, path='./plot.png', dpi=300):
    """plots images in a row in a single window. 

    Args:
        *imgs (tuple): Will be called with ax.imshow(imgs[i])
        colorbars (bool, optional): If True, all images will have a colorbar. Defaults to False.
        wait (bool, optional): If plot is not blocking will wait for buttonpress if True. Defaults to False.
        show (bool, optional): If True, the plot will block until window closed. Defaults to False.
    """
    n = len(imgs)
    fig, axes = plt.subplots(1, n, constrained_layout=True)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        if imgs[i].shape[-1] == 3:
            p = ax.imshow(imgs[i])
        else:
            p = ax.imshow(imgs[i], cmap='gray')
        ax.axis('off')
        if names is not None:
            ax.set_title(names[i], fontsize='small', fontname='serif')
        if colorbars:
            plt.colorbar(p, ax=ax)
    if save:
        # if one image save with plt.imsave
        if n == 1:
            if imgs[0].shape[-1] == 3:
                plt.imsave(path, imgs[0])
            else:
                plt.imsave(path, imgs[0], cmap='gray')
            return
        plt.savefig(path, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    elif wait:
        plt.show(block=False)
        plt.pause(.1)
        plt.waitforbuttonpress()

# plt.imshow(Image.open('./progress.gif').convert('RGB'))
# %%


@contextmanager
def ignore(*exceptions):
    try:
        yield
    except exceptions:
        pass
