# %% imports
from typing import Any, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from os.path import join
import logging
from utilities import *
from hornSchunck import horn_schunck, horn_schunck_multigrid, horn_schunck_withPDEloss

from flow_plot import colorplot_light
# %%


def PDE_loss(f1, f2, u, v, lapl_u, lapl_v, alphas):
    """
    Calculates Horn Schunck PDE
    """
    f_x = get_f_x(f1)
    f_y = get_f_y(f1)
    f_z = get_f_z(f1, f2)

    assert lapl_u.shape == f_x.shape
    assert u.unsqueeze(1).shape == f_x.shape
    assert v.unsqueeze(1).shape == f_x.shape
    assert f_z.unsqueeze(1).shape == f_x.shape

    first = f_x*(f_x*u.unsqueeze(1)+f_y*v.unsqueeze(1) +
                 f_z.unsqueeze(1))-alphas*lapl_u
    second = f_y*(f_x*u.unsqueeze(1)+f_y*v.unsqueeze(1) +
                  f_z.unsqueeze(1))-alphas*lapl_v

    return torch.sum(first**2)+torch.sum(second**2)


def Energy_loss(f1, f2, u, v, u_x, u_y, v_x, v_y, alphas):

    f_x = get_f_x(f1)
    f_y = get_f_y(f1)
    f_z = get_f_z(f1, f2)

    energy = (f_x*u+f_y*v+f_z)**2+alphas*(u_x**2+u_y**2+v_x**2+v_y**2)

    return torch.sum(energy**2)


def l2_nebenbed(img1, img2, img1_orig, img2_orig):
    """
    Return pertubation size in l2 norm
    """
    return torch.sum((img1-img1_orig)**2)+torch.sum((img2-img2_orig)**2)


def Relu_component(img1, img2, img1_orig, img2_orig, delta_max):
    """
    Delta_max is the maximum allowed pertubation in each pixel componentwise.
    returns reduce_sum( Relu(|delta|-delta_max) )
    """
    return torch.sum(torch.relu((img1-img1_orig).abs()-delta_max)) + torch.sum(torch.relu((img2-img2_orig).abs()-delta_max))


def Relu_globalaverage(img1, img2, img1_orig, img2_orig, delta_max):
    """
    Delta_max is the maximum allowed average image pertubation
    returns Relu( sum{i,j}{delta**2}/sum_{i,j}{1} - delta_max**2)
    """
    return torch.relu(torch.sum((img1-img1_orig)**2)/(img1.shape[0]*img1.shape[1]) - delta_max**2) + \
        torch.relu(torch.sum((img2-img2_orig)**2) /
                   (img2.shape[0]*img2.shape[1]) - delta_max**2)


def Relu_global(img1, img2, img1_orig, img2_orig, delta_max):
    """
    Delta_max is the maximum allowed image pertubation
    returns Relu( sum{i,j}{delta**2} - delta_max**2)
    """
    return torch.relu(torch.sum((img1-img1_orig)**2) - delta_max**2) + \
        torch.relu(torch.sum((img2-img2_orig)**2) - delta_max**2)

# %%


def adverserial_attack(img1, img2, img1_orig, img2_orig, target_u, target_v, delta_max=None, alphas=10., max_iter=100, opt=None, balance_factor=1., pertubation=None, nebenbed=Relu_component) -> Tuple[Any, Any, list]:
    """Executes adverserial attack. 
    Batched => shape: (batch,im_h,im_w)
    optional with L2 norm or upper limit to pertubation.
    optional with given pertubation or the image is the learnable part.

    Arguments:
        img1 {torch.tensor} -- image that gets changed
        img2 {torch.tensor} -- image that gets changed
        img1_orig {torch.tensor} -- original unchanged image
        img2_orig {torch.tensor} -- original unchanged image
        target_u {torch.tensor} -- flow target
        target_v {torch.tensor} -- flow target

    Keyword Arguments:
        delta_max {float} -- if given as float, the attack will have a upper limit to the pertubation (default: {None})
        alphas {torch.tensor} -- if given as a  (default: {10})
        max_iter {int} -- maximum number of optimisation iterations (default: {100})
        opt {Optimizer} -- Optimizer containing the learnable variables. If None, LGFGS is used to optimize img1, img2 or pertubation (default: {None})
        balance_factor {float} -- gets multiplied with the nebenbed (default: {1.})
        pertubation {torch.tensor} -- pertubation that gets added to the image every iteration. Optional, if not given img1,img2 should require gradients and will be changed. Shape:(2,im_h,im_w) or (2,batch_size,im_h,im_w). (default: {None})

    Returns:
        Tuple[Any,Any,list] -- _description_
    """
    if delta_max:
        # if there is a deltamax, define tolerance to check if target is reached
        tolerance = delta_max/2

    if isinstance(alphas, float):  # if single float change to batched alphas
        alphas = torch.ones((img1.shape[0], 1, 1, 1))*alphas

    logging.debug(
        f'Averserial Attack with alpha={alphas}, delta_max={delta_max}, max_iter={max_iter}')
    # if a pertubation is given, this pert. is the learnable part.
    if pertubation is not None:
        opt = opt or torch.optim.LBFGS([pertubation])
    else:
        opt = opt or torch.optim.LBFGS([img1, img2])

    lapl_u = get_lapl_f(target_u)
    lapl_v = get_lapl_f(target_v)
    loss_history = []
    best1 = best2 = img1.clone().to(device)
    bestloss = 1e10

    def closure(img1=img1, img2=img2):
        opt.zero_grad()

        # if a pertubation is given it has to be added to the image in every step (I think)
        if pertubation is not None:
            img1 = img1_orig+pertubation[0]
            img2 = img2_orig+pertubation[1]

        pde_loss = PDE_loss(img1, img2, target_u, target_v,
                            lapl_u, lapl_v, alphas)

        if delta_max:  # if you set a maximal condition
            neben_loss = nebenbed(img1, img2, img1_orig, img2_orig, delta_max)
        else:  # else do L2 condition
            neben_loss = l2_nebenbed(img1, img2, img1_orig, img2_orig)

        assert pde_loss.shape == neben_loss.shape, "shapes of losses dont match"
        loss = pde_loss+balance_factor*neben_loss
        logging.debug(f'pde_loss={pde_loss}, neben_loss={neben_loss}',)
        loss.backward()
        return loss

    for iteration in range(1, max_iter+1):

        l = closure()  # this is a duplicate for LBFGS Optimizer
        loss_history.append(l.cpu().detach().numpy())

        # if l!=l then l=nan -> break
        # if the last 10 iterations are equal stop the attack
        if l != l:
            if printing_enabled:
                print(f' Skipping next iterations\nloss {l} is nan\n')
            img1 = best1
            img2 = best2
            break

        if iteration > 10 and np.isclose(np.array(loss_history[-10:-1])-loss_history[-1], 0).all():
            if printing_enabled:
                print(f' Skipping next iterations\nloss {l} didnt change\n')
            break

        if max_iter > 25 and iteration % (max_iter//25) == 0:  # True:#
            progress_bar(iteration, max_iter, title='Adverserial',
                         msg=f'iter={iteration}, loss = {l}')

        opt.step(closure)

        # save the best results of the optimization. if the loss is NaN at the end,
        if delta_max and (Relu_component(img1, img2, img1_orig, img2_orig, delta_max) - delta_max).abs() < tolerance:
            if l < bestloss:
                bestloss = l
                best1 = img1.clone()
                best2 = img2.clone()
        else:
            bestloss = l
            best1 = img1.clone()
            best2 = img2.clone()

    if pertubation is not None:
        # print('returning pertubation')
        return pertubation, loss_history
    return img1, img2, loss_history


def adverserial_energy_attack(img1, img2, img1_orig, img2_orig, target_u, target_v, alpha=10, max_iter=100, opt=None, balance_factor=1.):
    opt = opt or torch.optim.LBFGS([img1, img2])

    u_x = get_f_x(target_u)
    u_y = get_f_y(target_u)
    v_x = get_f_x(target_v)
    v_y = get_f_y(target_v)

    for iteration in range(1, max_iter+1):
        def closure():
            opt.zero_grad()

            pde_loss = Energy_loss(img1, img2, target_u,
                                   target_v, u_x, u_y, v_x, v_y, alpha)
            neben_loss = l2_nebenbed(img1, img2, img1_orig, img2_orig)
            assert pde_loss.shape == neben_loss.shape, "shapes of losses dont match"
            loss = pde_loss+balance_factor*neben_loss
            logging.debug(f'pde_loss={pde_loss}, neben_loss={neben_loss}',)

            loss.backward()
            return loss

        l = closure()  # this is a duplicate for LBFGS Optimizer
        progress_bar(iteration, max_iter,
                     title='Adverserial Energy', msg=f'loss = {l}')

        opt.step(closure)

    return img1, img2


@convertTypes('tensor')
def full_attack(batch1_orig, batch2_orig, batch1=None, batch2=None, alphas=.1,
                # attack parameter
                target='zero', max_iter=100, balance_factor=1., delta_max=None, optimizer='LBFGS',
                max_iter_hornschunck=1000, flow_hs=None,  # horn schunck parameters
                show=False):
    """Generates the original Horn Schunck Flow, does the adverserial attack and calculates the Horn Schunck Flow from the pertubated image

    Args:
        batch1 (torch.tensor): batch of images to attack
        batch2 (torch.tensor): batch of images to attack
        img1_orig (torch.tensor): original image
        img2_orig (torch.tensor): original image
        alpha (torch.tensor, optional): smoothness factors (batched) of the Horn Schunck method. shape=(batch_size,1,1). Defaults to .1. if float, then expanded to required shape
        target (str, optional): Type of target. Options are: 'zero','inverse','original'. Defaults to 'zero'.
        max_iter (int, optional): maximum number of iterations for the adverserial attack. Defaults to 100.
        balance_factor (float, optional): gets multiplied with the limit of the pertubation size. Defaults to 1.0.
        delta_max (float, optional): maximum pertubation in the image. If None, the L2 loss is minimised instead of a hard limit. Defaults to None.
        optimizer (str, optional): String name of any optimizer within Torch, Aka 'Adam','LBFGS' or similar. A optimizer instance also works
        max_iter_hornschunck (int, optional): maximum number of iterations for each Horn Schunck Method. Defaults to 1000.
        flow_hs (np.ndarray[batch_size,im_h,im_w,2],optional): batch of horn schunck determined original flows  
        show (bool, optional): show all generated images. Defaults to False. TODO doesnt work with batch yet
    """
    if batch1 is None:
        batch1 = batch1_orig.clone().to(device)
        batch2 = batch2_orig.clone().to(device)

        batch1.requires_grad = True
        batch2.requires_grad = True

    batch_size, im_h, im_w = batch1_orig.shape[-3:]

    if isinstance(alphas, float):
        alphas = torch.ones((batch_size, 1, 1, 1))*alphas

    # get horn schunck solution of original image
    if flow_hs is None:
        u_orig = np.zeros(batch1_orig.shape)
        v_orig = np.zeros(batch1_orig.shape)
        for i, (img1, img2) in enumerate(zip(batch1_orig, batch2_orig)):
            u_orig[i], v_orig[i] = horn_schunck_multigrid(
                img1, img2, alpha=float(alphas[i]), max_iter=max_iter_hornschunck)
        u_orig = torch.from_numpy(u_orig).to(device)
        v_orig = torch.from_numpy(v_orig).to(device)

        flow_hs = torch.concat([u_orig.unsqueeze(-1), v_orig.unsqueeze(-1)])

    if show:
        show_images(u_orig, v_orig, colorplot_light(u_orig, v_orig), names=(
            'u_orig', 'v_orig', 'original flow',), colorbars=True)

    # create adverserial target
    if 'zero' in target.lower():
        target = torch.zeros((batch_size, im_h, im_w, 2)).to(device)
    elif 'inv' in target.lower():
        print('TODO: inv probably doesnt work with batch')
        u_orig = torch.tensor(u_orig, dtype=torch.float32).to(device)
        v_orig = torch.tensor(v_orig, dtype=torch.float32).to(device)
        target = torch.concat(
            (-u_orig.unsqueeze(-1), -v_orig.unsqueeze(-1)), dim=-1).to(device)
    elif 'orig' in target.lower():
        print('TODO: orig probably doesnt work with batch')
        u_orig = torch.tensor(u_orig, dtype=torch.float32).to(device)
        v_orig = torch.tensor(v_orig, dtype=torch.float32).to(device)
        target = torch.concat(
            (-u_orig.unsqueeze(-1), -v_orig.unsqueeze(-1)), dim=-1).to(device)

    if not isinstance(optimizer, torch.optim.Optimizer):
        optimizer = eval(f'torch.optim.{optimizer}([batch1,batch2])')

    # %% do the attack
    batch1, batch2, loss_hist = adverserial_attack(
        batch1, batch2, batch1_orig, batch2_orig, target[:, :, :, 0], target[:, :, :, 1], delta_max=delta_max, alphas=alphas, max_iter=max_iter, balance_factor=balance_factor)

    if show:
        show_images(batch1, batch1-batch1_orig, batch2, batch2-batch2_orig,
                    names=('img1', 'img1-orig1', 'img2', 'img2-orig2'), colorbars=True)

    flow_batch = []
    for img1, img2, alpha in zip(batch1, batch2, alphas):
        # %% get Horn Schunck result from these images
        u, v = horn_schunck_multigrid(
            img1, img2, alpha=alpha.squeeze(), max_iter=max_iter_hornschunck)
        u = torch.from_numpy(u).to(device)
        v = torch.from_numpy(v).to(device)

        if show:
            show_images(u, v, colorplot_light(u, v), names=(
                'u', 'v', 'pertubated flow',), colorbars=True)

            show_images(np.array(u)-np.array(u_orig), np.array(v)-np.array(v_orig), colorplot_light(u, v) -
                        colorplot_light(u_orig, v_orig), names=('difference u', 'difference v', 'difference flow'), colorbars=True)

        flow_batch.append(
            torch.cat((u.unsqueeze(-1), v.unsqueeze(-1)), dim=-1))

    flow_batch = torch.concat([x.unsqueeze(0) for x in flow_batch])
    return flow_hs, batch1, batch2, flow_batch, target, loss_hist


@convertTypes('tensor')
def get_metrics(pimg1, pimg2, img1, img2, pflow, hsflow, gtflow, target_flow, attack='l2'):
    """get all possible endpointerrors

    Arguments:
        pimg1 {torch.tensor} -- pertubated image
        pimg2 {torch.tensor} -- pertubated image
        img1 {torch.tensor} -- original image
        img2 {torch.tensor} -- original image
        pflow {torch.tensor} -- pertubated flow
        flow {torch.tensor} -- original flow
        gtflow {torch.tensor} -- ground truth flow
        type {str or float} -- type of attack. Options: float,'max' for $\delta_{max}$ attack else l2 attack

    Returns:
        epe {float} -- Average Endpointerror between pertubated and original flow
        epe_p_gt {float} -- Average Endpointerror between pertubated and ground truth flow
        delta {float} -- Size of the pertubation, type given as argument
    """
    epe = avg_EndPointError(pflow, hsflow)
    epe_p_gt = avg_EndPointError(pflow, gtflow)
    epe_target = avg_EndPointError(pflow, target_flow)

    if isinstance(attack, float) or attack.lower.contains('max'):
        delta = np.maximum(np.amax((img1-pimg1).abs().detach().cpu().numpy(), axis=(1, 2)),
                           np.amax((img2-pimg2).abs().detach().cpu().numpy(), axis=(1, 2)))  # np.maximum does elementwise max, np.amax along the axes
    else:
        delta = torch.norm(img1-pimg1, dim=(1, 2)) + \
            torch.norm(img2-pimg2, dim=(1, 2))
    return list(epe.detach().cpu().numpy()), list(epe_p_gt.detach().cpu().numpy()), list(delta), list(epe_target.detach().cpu().numpy())


@convertTypes('tensor')
def get_mse_metrics(pflow, hsflow, gtflow, target_flow, attack='l2'):
    """get all possible mse-related metrics

    Arguments:
        pimg1 {torch.tensor} -- pertubated image
        pimg2 {torch.tensor} -- pertubated image
        img1 {torch.tensor} -- original image
        img2 {torch.tensor} -- original image
        pflow {torch.tensor} -- pertubated flow
        flow {torch.tensor} -- original flow
        gtflow {torch.tensor} -- ground truth flow
        type {str or float} -- type of attack. Options: float,'max' for $\delta_{max}$ attack else l2 attack

    Returns:
        epe {float} -- Average Endpointerror between pertubated and original flow
        epe_p_gt {float} -- Average Endpointerror between pertubated and ground truth flow
        delta {float} -- Size of the pertubation, type given as argument
    """
    mse = torch.sum((pflow-hsflow)**2, dim=(1, 2, 3))
    mse_p_gt = torch.sum((pflow-gtflow)**2, dim=(1, 2, 3))
    mse_target = torch.sum((pflow-target_flow)**2, dim=(1, 2, 3))

    return list(mse.detach().cpu().numpy()), list(mse_p_gt.detach().cpu().numpy()), list(mse_target.detach().cpu().numpy())

# %% execution


def test_full_attack():
    image_folder = './dataset/RubberWhale/'
    img1_orig = torch.tensor(plt.imread(
        join(image_folder, 'frame10.png'))).mean(-1).to(device)  # mean for gray scale
    img2_orig = torch.tensor(plt.imread(
        join(image_folder, 'frame11.png'))).mean(-1).to(device)
    im_h, im_w = img1_orig.shape

    # "ground truth"
    u_gt, v_gt = torch.tensor(
        np.load(join(image_folder, 'flow.npy'))).to(device)
    u_gt[u_gt > 1e9] = 0
    v_gt[v_gt > 1e9] = 0

    # plot image and "ground truth"
    show_images(img1_orig, colorplot_light(u_gt, v_gt),
                names=("img2_orig", 'ground truth flow'))
    # %% for batch
    image_folder = './dataset/Dimetrodon/'
    more_img1_orig = torch.tensor(plt.imread(
        join(image_folder, 'frame10.png'))).mean(-1).to(device)  # mean for gray scale
    more_img2_orig = torch.tensor(plt.imread(
        join(image_folder, 'frame11.png'))).mean(-1).to(device)
    im_h, im_w = more_img1_orig.shape

    # "ground truth"
    u_gt, v_gt = torch.tensor(
        np.load(join(image_folder, 'flow.npy'))).to(device)
    u_gt[u_gt > 1e9] = 0
    v_gt[v_gt > 1e9] = 0

    # plot image and "ground truth"
    show_images(more_img1_orig, colorplot_light(u_gt, v_gt),
                names=("img2_orig", 'ground truth flow'))

    # %% create batch
    # more_img1_orig.unsqueeze(0)])
    batch1_orig = torch.concat([img1_orig.unsqueeze(0), ])
    # more_img2_orig.unsqueeze(0)])
    batch2_orig = torch.concat([img2_orig.unsqueeze(0), ])

    # %% adverserial attack parameter

    batch1 = batch1_orig.clone().detach().to(device)
    batch2 = batch2_orig.clone().detach().to(device)
    batch1.requires_grad = True
    batch2.requires_grad = True

    delta = .1
    _, batch1, batch2, flow_batch, _, _ = full_attack(
        batch1_orig, batch2_orig, batch1, batch2, delta_max=delta, max_iter=100, max_iter_hornschunck=1000)
    print((batch1[0]-batch1_orig[0]).max(), (batch2[0]-batch2_orig[0]).max())
    # print(delta,avg_EndPointError(flow,flow_orig))

    np.save('img1_batch.npy', batch1.detach().cpu().numpy()[0])
    np.save('img2_batch.npy', batch2.detach().cpu().numpy()[0])


def test_balancefactor():
    from datalogger import Logger

    ('/data/erik/mpi_sintel/training/final/shaman_2/frame_0004.png',
     '/data/erik/mpi_sintel/training/final/shaman_2/frame_0005.png')

    img1_orig = torch.tensor(plt.imread(
        '/data/erik/mpi_sintel/training/final/shaman_2/frame_0004.png')).mean(-1).to(device)  # mean for gray scale
    img2_orig = torch.tensor(plt.imread(
        '/data/erik/mpi_sintel/training/final/shaman_2/frame_0005.png')).mean(-1).to(device)
    im_h, im_w = img1_orig.shape

    # "ground truth"
    from flow_IO import readFloFlow
    flow_gt = torch.tensor(readFloFlow(
        '/data/erik/mpi_sintel/training/flow/shaman_2/frame_0004.flo')).to(device)
    u_gt, v_gt = flow_gt[:, :, 0], flow_gt[:, :, 1]
    u_gt[u_gt > 1e9] = 0
    v_gt[v_gt > 1e9] = 0

    # plot image and "ground truth"
    show_images(img1_orig, colorplot_light(u_gt, v_gt),
                names=("img2_orig", 'ground truth flow'))

    # %% adverserial attack parameter

    img1 = img1_orig.clone().detach().to(device)
    img2 = img2_orig.clone().detach().to(device)

    img1 = torch.rand_like(img1_orig).to(device)
    img2 = torch.rand_like(img2_orig).to(device)

    target = torch.zeros((1, 2, im_h, im_w)).to(device)

    img1.requires_grad = True
    img2.requires_grad = True

    max_iter = 1000
    max_iter_hornschunck = 1000

    alpha = .1
    balance_factor = 0.0001  # gets multiplied with norm of pertubation
    delta_max = 1.8358e-05

    logger = Logger('./balancefactor.json')
    logger['delta_balance'] = {1.8358e-05: 0.0001}

    opt = torch.optim.LBFGS([img1, img2])

    with logger:
        for delta_max in 1/np.logspace(0, 5, 20, dtype=float):
            balance_factor = 1e5
            loss_hist = [np.nan]
            n = 0
            while True:

                img1 = torch.rand_like(img1_orig).to(device)
                img2 = torch.rand_like(img2_orig).to(device)
                img1.requires_grad = True
                img2.requires_grad = True
                opt = torch.optim.LBFGS([img1, img2])

                _, _, loss_hist = adverserial_attack(img1.unsqueeze(0), img2.unsqueeze(0), img1_orig.unsqueeze(0), img2_orig.unsqueeze(
                    0), target[:, 0], target[:, 1], delta_max=delta_max, alphas=alpha, max_iter=max_iter, balance_factor=balance_factor, opt=opt)
                if np.isnan(loss_hist[-1]):
                    print(balance_factor, "didn't work")
                    balance_factor /= 10.
                else:
                    print(balance_factor, "WORKED WITHOUT NAN")
                    n += 1
                    if n > 5:
                        logger["delta_balance"][delta_max] = balance_factor
                        logger.toFile()
                        break


def test_adverserialAttack():

    img1_path, img2_path = ('/data/erik/mpi_sintel/training/final/shaman_2/frame_0004.png',
                            '/data/erik/mpi_sintel/training/final/shaman_2/frame_0005.png')
    flow_path = '/data/erik/mpi_sintel/training/flow/shaman_2/frame_0004.flo'
    # img1_path,img2_path=('./dataset/mpi_sintel/training/final/shaman_2/frame_0004.png', './dataset/mpi_sintel/training/final/shaman_2/frame_0005.png')
    # flow_path='./dataset/mpi_sintel/training/flow/shaman_2/frame_0004.flo'

    img1_orig = torch.tensor(plt.imread(
        img1_path)).mean(-1).to(device)  # mean for gray scale
    img2_orig = torch.tensor(plt.imread(img2_path)).mean(-1).to(device)
    im_h, im_w = img1_orig.shape

    # "ground truth"
    from flow_IO import readFloFlow
    flow_gt = torch.tensor(readFloFlow(flow_path)).to(device)
    u_gt, v_gt = flow_gt[:, :, 0], flow_gt[:, :, 1]
    u_gt[u_gt > 1e9] = 0
    v_gt[v_gt > 1e9] = 0

    # plot image and "ground truth"
    show_images(img1_orig, colorplot_light(u_gt, v_gt),
                names=("img2_orig", 'ground truth flow'))

    # %% adverserial attack parameter

    img1 = img1_orig.clone().detach().to(device)
    img2 = img2_orig.clone().detach().to(device)

    # img1 = torch.rand_like(img1_orig).to(device)
    # img2 = torch.rand_like(img2_orig).to(device)

    target = torch.zeros((2, im_h, im_w)).to(device)

    img1.requires_grad = True
    img2.requires_grad = True

    max_iter = 1000
    max_iter_hornschunck = 1000

    alpha = .1
    balance_factor = 1.  # gets multiplied with norm of pertubation
    delta_max = 1.8358e-05

    opt = torch.optim.Adam([img1, img2])

    # %% get horn schunck solution of original image

    u_orig, v_orig = horn_schunck_multigrid(
        img1_orig, img2_orig, alpha=alpha, max_iter=max_iter_hornschunck)

    show_images(u_orig, v_orig, colorplot_light(u_orig, v_orig), names=(
        'u_orig', 'v_orig', 'original flow',), colorbars=True)

    # if we want the inverse of the original flow
    # u_orig = torch.tensor(u_orig, dtype=torch.float32).to(device)
    # v_orig = torch.tensor(v_orig, dtype=torch.float32).to(device)
    # target = torch.concat((-u_orig.unsqueeze(0),-v_orig.unsqueeze(0)),dim=0).to(device)

    # %% do the attack
    img1, img2, loss_hist = adverserial_attack(img1.unsqueeze(0), img2.unsqueeze(0), img1_orig.unsqueeze(0), img2_orig.unsqueeze(0), target[0].unsqueeze(
        0), target[1].unsqueeze(0), delta_max=delta_max, alphas=alpha, max_iter=max_iter, balance_factor=balance_factor, opt=opt, nebenbed=Relu_global)

    # %% show results
    show_images(img1, img1-img1_orig, img2, img2-img2_orig,
                names=('img1', 'img1-orig1', 'img2', 'img2-orig2'), colorbars=True)

    # %% get Horn Schunck result from these images
    u, v = horn_schunck_multigrid(
        img1, img2, alpha=alpha, max_iter=max_iter_hornschunck)
    show_images(u, v, colorplot_light(u, v), names=(
        'u', 'v', 'pertubated flow',), colorbars=True)

    show_images(np.array(u)-np.array(u_orig), np.array(v)-np.array(v_orig), colorplot_light(u, v) -
                colorplot_light(u_orig, v_orig), names=('difference u', 'difference v', 'difference flow'), colorbars=True)
    print()

    print('Horn Schunck vs Pertubated Horn Schunck')
    print(f'||u_orig-u||        = {np.linalg.norm(u_orig-u)}')
    print(f'||v_orig-v||        = {np.linalg.norm(v_orig-v)}')
    print()

    # print('Ground Truth vs Pertubated Horn Schunck')
    # print(f'||u_gt-u||          = {np.linalg.norm(u_gt-u)}')
    # print(f'||v_gt-v||          = {np.linalg.norm(v_gt-v)}')
    # print()

    # print('Horn Schunck vs Ground Truth')
    # print(f'||u_orig-u_gt||     = {np.linalg.norm(u_orig-u_gt.detach().numpy())}')
    # print(f'||v_orig-v_gt||     = {np.linalg.norm(v_orig-v_gt.detach().numpy())}')
    # print()

    print('Pertubated vs original image')
    print(f'||img1-img1_orig||  = {torch.linalg.norm(img1-img1_orig)}')
    print(f'||img2-img2_orig||  = {torch.linalg.norm(img2-img2_orig)}')
    print()

    print('Biggest pertubation')
    print(f'max(|img1-img1_orig|)  = {(img1-img1_orig).abs().max()}')
    print(f'max(|img2-img2_orig|)  = {(img2-img2_orig).abs().max()}')
    print()

    plt.show()


if __name__ == "__main__":
    # torch.autograd.anomaly_mode.set_detect_anomaly(True)
    test_adverserialAttack()

# %%
