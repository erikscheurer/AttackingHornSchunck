import torch
import matplotlib.pyplot as plt
from hornSchunck import horn_schunck
from utilities import *
from adverserial import adverserial_attack, adverserial_energy_attack
from os.path import join

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.debug('debug')
    logging.info('info')

# %% Read images (in black and white)
    image_folder = './dataset/RubberWhale/'
    img1_orig = torch.tensor(plt.imread(
        join(image_folder, 'frame10.png'))).mean(-1)  # mean for gray scale
    img2_orig = torch.tensor(plt.imread(
        join(image_folder, 'frame11.png'))).mean(-1)
    im_h, im_w = img1_orig.shape

    # "ground truth"
    u, v = np.load(join(image_folder, 'flow.npy'))
    u[u > 1e9] = 0
    v[v > 1e9] = 0

    # plot image and "ground truth"
    show_images(img1_orig, u, v, colorbars=True)

    alpha = .1
    max_iter_hornschunck = 1000

    # %% get horn schunck solution of original image
    u_orig, v_orig = horn_schunck(
        img1_orig, img2_orig, alpha=alpha, max_iter=max_iter_hornschunck)

    # if we want the inverse of the original flow
    u_orig = torch.tensor(u_orig, dtype=torch.float32)
    v_orig = torch.tensor(v_orig, dtype=torch.float32)
    target = torch.concat((-u_orig.unsqueeze(0), -v_orig.unsqueeze(0)), dim=0)
    target = torch.ones((2, im_h, im_w))

    prev1 = img1_orig.clone()
    prev2 = img2_orig.clone()
    prev1e = img1_orig.clone()
    prev2e = img2_orig.clone()

# %% Do both attacks two times
    for i in range(3):
        # %% PDE adverserial attack parameter
        img1 = torch.rand_like(img1_orig)
        img2 = torch.rand_like(img2_orig)

        img1.requires_grad = True
        img2.requires_grad = True

        max_iter = 100

        opt = torch.optim.LBFGS([img1, img2], lr=.01)

    # %% PDE attack
        img1, img2 = adverserial_attack(
            img1, img2, img1_orig, img2_orig, target[0], target[1], alpha=alpha, max_iter=max_iter, opt=opt)

    # %% show results
        img1 = img1.cpu().detach()
        img2 = img2.cpu().detach()

        print('difference to before:', (img1-prev1).abs().sum())
        print('difference to before:', (img2-prev2).abs().sum())

        show_images(img1-prev1, img2-prev2,
                    names=('diff to prev 1', 'diff to prev 2'), colorbars=True)

        prev1 = img1.clone().detach()
        prev2 = img2.clone().detach()

    # %% Energy attack parameter
        img1e = torch.rand_like(img1_orig)
        img2e = torch.rand_like(img2_orig)

        img1e.requires_grad = True
        img2e.requires_grad = True

        max_iter = 100

        opt = torch.optim.LBFGS([img1e, img2e], lr=.01)

    # %% do the Energy attack
        img1e, img2e = adverserial_energy_attack(
            img1e, img2e, img1_orig, img2_orig, target[0], target[1], alpha=alpha, max_iter=max_iter, opt=opt)

    # %% show results
        img1e = img1e.cpu().detach()
        img2e = img2e.cpu().detach()

        print('difference to before:', (img1e-prev1e).abs().sum())
        print('difference to before:', (img2e-prev2e).abs().sum())

        show_images(img1e-prev1e, img2e-prev2e,
                    names=('diff to prev 1e', 'diff to prev 2e'), colorbars=True)

        prev1e = img1e.clone().detach()
        prev2e = img2e.clone().detach()

        print('difference between Energy and PDE attack:',
              (img1e-img1).abs().sum())
        print('difference between Energy and PDE attack:',
              (img2e-img2).abs().sum())

    plt.show()
# %%
