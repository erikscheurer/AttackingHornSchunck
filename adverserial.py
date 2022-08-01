# %% imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from flow_IO import readFloFlow
from utilities import *
from hornSchunck import horn_schunck_multigrid

from flow_plot import colorplot_light
# %%


@torch.jit.script
def PDE_loss(f1, f2, u, v, lapl_u, lapl_v, alpha: float):
    """
    Calculates Horn Schunck PDE
    """
    f_x = jit_fx(f1)[0, 0]
    f_y = jit_fy(f1)[0, 0]
    f_z = jit_fz(f1, f2)

    assert lapl_u.shape == f_x.shape
    assert u.shape == f_x.shape
    assert v.shape == f_x.shape
    assert f_z.shape == f_x.shape

    first = f_x*(f_x*u+f_y*v+f_z)-alpha*lapl_u
    second = f_y*(f_x*u+f_y*v+f_z)-alpha*lapl_v

    return torch.sum(first**2)+torch.sum(second**2)


@torch.jit.script
def Energy_loss(f1, f2, u, v, u_x, u_y, v_x, v_y, alpha: float):

    f_x = jit_fx(f1)
    f_y = jit_fy(f1)
    f_z = jit_fz(f1, f2)

    energy = (f_x*u+f_y*v+f_z)**2+alpha*(u_x**2+u_y**2+v_x**2+v_y**2)

    return torch.sum(energy)


@torch.jit.script
def l2_nebenbed(img1, img2, img1_orig, img2_orig, delta_max: float = 0):
    """
    Return pertubation size in l2 norm. delta_max is unused just to make the implementation easier
    """
    return torch.sum((img1-img1_orig)**2)+torch.sum((img2-img2_orig)**2)


@torch.jit.script
def Relu_component(img1, img2, img1_orig, img2_orig, delta_max: float):
    """
    Delta_max is the maximum allowed pertubation in each pixel componentwise.
    returns reduce_sum( Relu(|delta|-delta_max) )
    """
    return torch.sum(torch.relu((img1-img1_orig).abs()-delta_max)) + torch.sum(torch.relu((img2-img2_orig).abs()-delta_max))


@torch.jit.script
def Relu_component_avg(img1, img2, img1_orig, img2_orig, delta_max: float):
    """
    Delta_max is the maximum allowed pertubation in each pixel componentwise.
    returns reduce_sum( Relu(|delta|-delta_max) )/sqrt(N)
    """
    return (Relu_component(img1, img2, img1_orig, img2_orig, delta_max))/torch.sqrt(img1.shape[0]*img1.shape[1])


@torch.jit.script
def global_l2norm(delta):
    return torch.norm(delta)
    # return torch.sum(delta**2)


@torch.jit.script
def global_l2average(delta):
    return global_l2norm(delta)/torch.sqrt(delta.shape[0]*delta.shape[1])


@torch.jit.script
def Relu_globalaverage(img1, img2, img1_orig, img2_orig, delta_max: float):
    """
    returns loss for global average delta
    Delta_max is the maximum allowed average image pertubation
    returns Relu( sum{i,j}{delta**2}/sum_{i,j}{1} - delta_max**2)
    """
    bothimgs = torch.concat((img1, img2))
    bothimgs_orig = torch.concat((img1_orig, img2_orig))
    return torch.relu(global_l2average(bothimgs-bothimgs_orig)**2 - delta_max**2)


@torch.jit.script
def Relu_global(img1, img2, img1_orig, img2_orig, delta_max: float):
    """
    returns for global delta
    Delta_max is the maximum allowed image pertubation
    returns Relu( sum{i,j}{delta**2} - delta_max**2)
    """
    return torch.relu(global_l2norm(img1-img1_orig)**2 - delta_max**2) + \
        torch.relu(global_l2norm(img2-img2_orig)**2 - delta_max**2)


@torch.jit.script
def changeOfVariable(toOptimize, tol: float = 1e-6):
    """
    returns the image in range $[0,1]$ when given the optimized image in range $(-infty,infty)$
    \delta = .5(tanh(w)+1)-x
    => \delta+x = .5(tanh(w)+1)
    => here: constrained_img = changeOfVariable(trainable_img)
    """
    # if tol == 0 and (toOptimize.min() == -torch.inf or toOptimize.max() == torch.inf):
    #     logging.warning("toOptimize contains -inf or inf")
    return .5*((1-tol)*torch.tanh(toOptimize)+1)


def invChangeOfVariable(img, tol=1e-6):
    """
    returns the optimized image in range $(-infty,infty)$ when given the orignal image in range $[0,1]$
    If passed img created from changeOfVariable, can pass tol=0, because the img already doesn't have 0 and 1 in it

    """
    if tol == 0 and (img.min() <= 0 or img.max() >= 1):
        logging.warning(
            "img contains 0 or 1 which will beconverted to -inf or inf")
    return torch.atanh((2*img-1)/(1+tol))


def changeOfVariable_pertubation(toOptimize, orig_image, tol=1e-6):
    """
    \delta = .5(tanh(trainable_pert)+1)-original_img
    """
    if tol == 0 and (toOptimize.min() == -torch.inf or toOptimize.max() == torch.inf):
        logging.warning("toOptimize contains -inf or inf")
    return .5*((1-tol)*torch.tanh(toOptimize)+1)-orig_image


def invChangeOfVariable_pertubation(pertubation, orig_image, tol=1e-6):
    """
    returns the optimized pertubation in range $[-infty,infty]$ when given the orignal pertubation+img in range $[0,1]$
    If passed img created from changeOfVariable, can pass tol=0, because the img already doesn't have 0 and 1 in it
    """
    if tol == 0 and ((pertubation+orig_image).min() <= 0 or (pertubation+orig_image).max() >= 1):
        logging.warning(
            "img contains 0 or 1 which will beconverted to -inf or inf")

    return torch.atanh((2*(pertubation+orig_image)-1)/(1+tol))

# %%


def adverserial_attack(img1, img2, img1_orig, img2_orig, target_u, target_v, delta_max, alpha=.1, max_iter=100, opt=None, scheduler=None, balance_factor=1.e8, pde_type='energy', nebenbed=Relu_globalaverage, ChangeOfVar=True, return_last_iter=False):
    """Executes adverserial attack.
    optional with L2 norm or upper limit to pertubation.
    optional with given pertubation or the image is the learnable part.
    Change of variable is implemented, that img1,img2 become the hidden variables and the new introduces variables are the constrained images. the return imgs go also through a change of variable.

    Arguments:
        img1 {torch.tensor} -- image that gets changed
        img2 {torch.tensor} -- image that gets changed
        img1_orig {torch.tensor} -- original unchanged image
        img2_orig {torch.tensor} -- original unchanged image
        target_u {torch.tensor} -- flow target
        target_v {torch.tensor} -- flow target
        delta_max {float} -- the attack will have a upper limit to the pertubation)

    Keyword Arguments:
        alphas {torch.tensor} -- if given as a  (default: {.1})
        max_iter {int} -- maximum number of optimisation iterations (default: {100})
        opt {Optimizer} -- Optimizer containing the learnable variables. If None, LGFGS is used to optimize img1, img2 or pertubation (default: {None})
        balance_factor {float} -- gets multiplied with the nebenbed (default: {1.})
        ChangeOfVar {bool} -- Toggles change of Variables for the optimizer. This constrains the resulting image to a range of [0,1].
        return_last_iter {bool} -- if True, only the last iteration is returned instead of the best (default: {False})

    Returns:
        Tuple[Any,Any,list] -- img1, img2, loss_history
        Tuple[Any,list] -- pertubation, loss_history
    """
    from utilities import device
    tolerance = 1e-10  # define tolerance to check if target is reached or if loss doesnt change

    opt = opt or torch.optim.LBFGS([img1, img2])

    if pde_type == 'pde':
        lapl_u = get_lapl_f(target_u).squeeze()
        lapl_v = get_lapl_f(target_v).squeeze()
    else:
        u_x = get_f_x(target_u)
        u_y = get_f_y(target_u)
        v_x = get_f_x(target_v)
        v_y = get_f_y(target_v)

    loss_history = [[], [], []]
    best1 = best2 = img1.clone().to(device)
    bestloss = 1e10

    def closure(img1=img1, img2=img2, return_seperated=False):
        """
        Keyword Arguments:
            img1 {torch.tensor} -- img that is in the optimizer (default: {img1})
            img2 {torch.tensor} -- img that is in the optimizer (default: {img2})
            return_seperated {bool} -- if true return loss, pde-loss, neben-loss else only loss (default: {False})
        """
        # assertion that the used imgs are the imgs in the optimizer. can become a problem with clipping for example
        assert id(img1) == id(
            opt.param_groups[0]['params'][0]), "Img1 not in parameter group of optimizer"
        assert id(img2) == id(
            opt.param_groups[0]['params'][1]), "Img2 not in parameter group of optimizer"

        opt.zero_grad()

        if ChangeOfVar:
            f1 = changeOfVariable(img1)  # working image
            f2 = changeOfVariable(img2)
            assert (f1 <= 1.).all() and (
                0. <= f1).all(), "Change of Variable didnt work"
            assert (f2 <= 1.).all() and (
                0. <= f2).all(), "Change of Variable didnt work"
        else:
            f1 = img1  # without change of variable just work with original
            f2 = img2

        neben_loss = nebenbed(f1, f2, img1_orig, img2_orig, delta_max*.999)

        if False:  # not return_seperated and neben_loss > 0:
            loss = neben_loss*balance_factor
        else:
            if pde_type == 'pde':
                pde_loss = PDE_loss(f1, f2, target_u, target_v,
                                    lapl_u, lapl_v, alpha)
                assert pde_loss.shape == neben_loss.shape, "shapes of losses dont match"
                loss = pde_loss + balance_factor * neben_loss
            else:
                pde_loss = Energy_loss(f1, f2, target_u, target_v,
                                       u_x, u_y, v_x, v_y, alpha)

                assert pde_loss.shape == neben_loss.shape, "shapes of losses dont match"
                loss = pde_loss + balance_factor * neben_loss

            # loss = pde_loss/10**max(pde_loss.detach(), torch.tensor(1e-11)).log10().round() + \
            #     balance_factor * neben_loss / \
            #     10**max(neben_loss.detach(),
            #             torch.tensor(1e-11)).log10().round()

        if return_seperated:
            return loss, pde_loss, neben_loss

        loss.backward()
        return loss

    l, pde, neben = closure(img1, img2, return_seperated=True)
    loss_history[0].append(l.detach().cpu().numpy())
    loss_history[1].append(pde.detach().cpu().numpy())
    loss_history[2].append(neben.detach().cpu().numpy())

    found_good_delta = False
    delta_list = []
    for iteration in range(1, max_iter+1):

        opt.step(closure)

        # this is a duplicate when using LBFGS Optimizer
        l, pde, neben = closure(img1, img2, return_seperated=True)

        if scheduler:
            scheduler.step()

        loss_history[0].append(l.detach().cpu().numpy())
        loss_history[1].append(pde.detach().cpu().numpy())
        loss_history[2].append(neben.detach().cpu().numpy())

        # if l!=l then l=nan -> break
        if l != l:
            progress_bar(iteration, max_iter, title='Adverserial',
                         msg=f'iter={iteration}, pde-loss={pde:.04}, neben-loss={neben:.04}, constraint={found_good_delta}')
            if printing_enabled:
                print(f' Skipping next iterations\nloss {l} is nan\n')
            break

        # check if change is small, if yes, break
        if ChangeOfVar:
            globalnorm, globalavg, absolute, l2, maximum = get_deltas(
                changeOfVariable(img1), changeOfVariable(img2), img1_orig, img2_orig)
        else:
            globalnorm, globalavg, absolute, l2, maximum = get_deltas(
                img1, img2, img1_orig, img2_orig)

        delta_list.append(globalavg.detach().cpu().numpy())
        # checks if the pertubation doesn't change for 10 steps. If yes, break
        # if iteration > 10 and np.isclose(np.array(delta_list[-10:-1])-delta_list[-1], 0, atol=delta_max*.0001).all():
        # checks if the loss 'l' didn't change for the last 10 steps. If yes, break
        if iteration > 10 and np.isclose(np.array(loss_history[0][-10:-1]), loss_history[0][-1], atol=tolerance).all():
            progress_bar(iteration, max_iter, title='Adverserial',
                         msg=f'iter={iteration}, pde-loss={pde:.04}, neben-loss={neben:.04}, delta={delta_list[-1]:.04}, constraint={found_good_delta}')
            if printing_enabled:
                print(f' Skipping next iterations\nloss {l} didnt change\n')
            if scheduler is not None:
                print('last lr', scheduler.get_last_lr())
            break

        # progressbar
        if max_iter > 25 and iteration % (max_iter//25) == 0:  # True:  #
            progress_bar(iteration, max_iter, title='Adverserial',
                         msg=f'iter={iteration}, pde-loss={pde:.04}, neben-loss={neben:.04}, delta={delta_list[-1]:.04}, constraint={found_good_delta}')

        # save the best results of the optimization. if the loss is NaN at the end, return this backup
        if ChangeOfVar:
            constraint_fulfilled = nebenbed(changeOfVariable(img1), changeOfVariable(img2),
                                            img1_orig, img2_orig, delta_max) < tolerance
        else:
            constraint_fulfilled = nebenbed(img1, img2,
                                            img1_orig, img2_orig, delta_max) < tolerance

        # dont check in the first few iterations, because we may initialize as 0:
        # if either there has not been a delta that fulfilles the constraint or if this iteration also fulfilles the constraint
        if iteration > 9 and (not found_good_delta or constraint_fulfilled):

            # check if the loss is better than before. If the constraint is fulfilled for the first time, this automatically qualifies to save as the current best.
            if l < bestloss or (not found_good_delta and constraint_fulfilled):
                # if fulfilled for the first time, then set the found_variable as true
                if (not found_good_delta and constraint_fulfilled):
                    found_good_delta = True

                bestloss = l
                best1 = img1.clone()
                best2 = img2.clone()

    if not return_last_iter:
        # only return the best results
        img1 = best1
        img2 = best2
    loss_history = [[float(l) for l in losstype]
                    for losstype in loss_history]

    if ChangeOfVar:
        return changeOfVariable(img1), changeOfVariable(img2), loss_history, found_good_delta
    return img1, img2, loss_history, found_good_delta


def full_attack(img1_orig, img2_orig, alpha=.1, scheduler=None, schedule_gamma=.999,  # attack parameter:
                target='zero', max_iter=1000, balance_factor=1.e8, delta_max=None, optimizer='LBFGS', opt_args={"max_iter": 10, "history_size": 20}, nebenbed=Relu_globalaverage, pde_type='pde', zero_init=True,
                max_iter_hornschunck=1000, flow_hs=None,  # horn schunck parameters
                show=False, changeOfVar=True, return_last_iter=False):
    """Generates the original Horn Schunck Flow, does the adverserial attack and calculates the Horn Schunck Flow from the perturbed image

    Args:
        img1_orig (torch.tensor): original image
        img2_orig (torch.tensor): original image
        img1 (torch.tensor,optional): image to attack
        img2 (torch.tensor,optional): image to attack
        alpha (torch.tensor, optional): smoothness factor of the Horn Schunck method
        target (str, optional): Type of target. Options are: 'zero','inverse','original' or any float. Defaults to 'zero'.
        max_iter (int, optional): maximum number of iterations for the adverserial attack. Defaults to 100.
        balance_factor (float, optional): gets multiplied with the limit of the pertubation size. Defaults to 1.0.
        delta_max (float, optional): maximum pertubation in the image. If None, the L2 loss is minimised instead of a hard limit. Defaults to None.
        optimizer (str, optional): String name of any optimizer within torch, Aka 'Adam','LBFGS' or similar. A optimizer instance also works
        max_iter_hornschunck (int, optional): maximum number of iterations for each Horn Schunck Method. Defaults to 1000.
        flow_hs (np.ndarray[batch_size,im_h,im_w,2],optional): batch of horn schunck determined original flows
        show (bool, optional): show all generated images. Defaults to False.
    """
    from utilities import device
    if zero_init:
        img1 = img1_orig.clone().to(device)
        img2 = img2_orig.clone().to(device)
    else:
        img1 = torch.rand_like(img1_orig).to(device)
        img2 = torch.rand_like(img2_orig).to(device)

    # get horn schunck solution of original image
    if flow_hs is None:
        u_orig, v_orig = horn_schunck_multigrid(
            img1_orig, img2_orig, alpha=alpha, max_iter=max_iter_hornschunck)

        u_orig = torch.from_numpy(u_orig).to(device)
        v_orig = torch.from_numpy(v_orig).to(device)
        flow_hs = flowFromUV(u_orig, v_orig).float()

    if show:
        u_orig, v_orig = flow_hs[:, :, 0], flow_hs[:, :, 1]
        show_images(u_orig, v_orig, colorplot_light(u_orig, v_orig),
                    names=('u_orig', 'v_orig', 'original flow',), colorbars=False)

    # create adverserial target
    if isNumber(target):
        target = torch.ones_like(flow_hs).to(device).float() * float(target)
    elif 'zero' in target.lower():
        target = torch.zeros_like(flow_hs).to(device).float()
    elif 'inv' in target.lower():
        target = -flow_hs.float()
    elif 'orig' in target.lower():
        u_orig = torch.tensor(u_orig, dtype=torch.float32).to(device)
        v_orig = torch.tensor(v_orig, dtype=torch.float32).to(device)
        target = torch.concat(
            (-u_orig.unsqueeze(-1), -v_orig.unsqueeze(-1)), dim=-1).to(device)

    if changeOfVar:
        img1 = invChangeOfVariable(img1)
        img2 = invChangeOfVariable(img2)
    img1 = img1.requires_grad_(True)
    img2 = img2.requires_grad_(True)

    # initialize Optimizer
    if not isinstance(optimizer, torch.optim.Optimizer):
        optimizer = eval(f'torch.optim.{optimizer}([img1,img2],**opt_args)')
    if scheduler and 'exp' in scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, schedule_gamma)

    # %% do the attack
    img1, img2, loss_hist, found_good_delta = adverserial_attack(
        img1, img2, img1_orig, img2_orig, target[:, :, 0], target[:, :, 1], delta_max=delta_max, alpha=alpha, max_iter=max_iter, opt=optimizer, scheduler=scheduler, balance_factor=balance_factor, nebenbed=nebenbed, pde_type=pde_type, ChangeOfVar=changeOfVar, return_last_iter=return_last_iter)

    if show:
        show_images(img1, img1-img1_orig, img2, img2-img2_orig,
                    names=('img1', 'img1-orig1', 'img2', 'img2-orig2'), colorbars=False)

    # %% get Horn Schunck result from these images
    u, v = horn_schunck_multigrid(img1.detach().cpu().numpy(), img2.detach(
    ).cpu().numpy(), alpha=alpha, max_iter=max_iter_hornschunck)
    u = torch.from_numpy(u).to(device)
    v = torch.from_numpy(v).to(device)

    if show:
        show_images(u, v, colorplot_light(u, v), names=(
            'u', 'v', 'perturbed flow',), colorbars=False)

        show_images(u-u_orig, v-v_orig, colorplot_light(u-u_orig, v-v_orig),
                    names=('difference u', 'difference v', 'difference flow'), colorbars=False)
    flow_perturbed = flowFromUV(u, v)
    return flow_hs, img1, img2, flow_perturbed, target, loss_hist, found_good_delta


@convertTypes('tensor')
def get_deltas(pimg1, pimg2, img1, img2):
    """
    get different delta metrics
    returns globalnorm, globalavg, absolute, l2, maximum
    """
    pbothimgs = torch.concat((pimg1, pimg2))
    bothimgs = torch.concat((img1, img2))

    # compute all delta metrics
    globalnorm = global_l2norm(pbothimgs-bothimgs)
    globalavg = global_l2average(pbothimgs-bothimgs)
    absolute = torch.sum((pbothimgs-bothimgs).abs())
    maximum = (pbothimgs-bothimgs).max()
    l2 = l2_nebenbed(pimg1, pimg2, img1, img2)

    return globalnorm, globalavg, absolute, l2, maximum


@convertTypes('tensor')
def get_metrics(pimg1, pimg2, img1, img2, pflow, hsflow, gtflow, target_flow):
    """get all possible endpointerrors

    Arguments:
        pimg1 {torch.tensor} -- perturbed image
        pimg2 {torch.tensor} -- perturbed image
        img1 {torch.tensor} -- original image
        img2 {torch.tensor} -- original image
        pflow {torch.tensor} -- perturbed flow
        flow {torch.tensor} -- original flow
        gtflow {torch.tensor} -- ground truth flow
        type {str or float} -- type of attack. Options: float,'max' for $\delta_{max}$ attack else l2 attack

    Returns:
        epe {float} -- Average endpointerror between initial and perturbed flow
        epe_p_gt {float} -- Average endpointerror between perturbed and ground truth flow
        epe_target {float} -- Average endpointerror between perturbed and target flow
        epe_gt_target {float} -- Average endpointerror between ground truth and target flow
        epe_orig_target {float} -- Average endpointerror between original and target flow
    """
    epe = avg_EndPointError(pflow, hsflow)
    if gtflow is not None:
        epe_p_gt = avg_EndPointError(pflow, gtflow)
        epe_p_gt = float(epe_p_gt.detach().cpu().numpy())
        epe_gt_target = avg_EndPointError(gtflow, target_flow)
        epe_gt_target = float(epe_gt_target.detach().cpu().numpy())
    else:
        epe_p_gt = None
        epe_gt_target = None
    epe_target = avg_EndPointError(pflow, target_flow)
    epe_orig_target = avg_EndPointError(hsflow, target_flow)

    return float(epe.detach().cpu().numpy()), epe_p_gt, float(epe_target.detach().cpu().numpy()), epe_gt_target, float(epe_orig_target.detach().cpu().numpy())


@convertTypes('tensor')
def get_mse_metrics(pflow, hsflow, gtflow, target_flow):
    """get all possible mse-related metrics

    Arguments:
        pimg1 {torch.tensor} -- perturbed image
        pimg2 {torch.tensor} -- perturbed image
        img1 {torch.tensor} -- original image
        img2 {torch.tensor} -- original image
        pflow {torch.tensor} -- perturbed flow
        flow {torch.tensor} -- original flow
        gtflow {torch.tensor} -- ground truth flow
        type {str or float} -- type of attack. Options: float,'max' for $\delta_{max}$ attack else l2 attack
    """
    mse = torch.sum((pflow-hsflow)**2)
    if gtflow is not None:  # for test set
        mse_p_gt = torch.sum((pflow-gtflow)**2)
        mse_p_gt = float(mse_p_gt.detach().cpu().numpy())
        mse_gt_target = torch.sum((gtflow-target_flow)**2)
        mse_gt_target = float(mse_gt_target.detach().cpu().numpy())
    else:
        mse_p_gt = None
        mse_gt_target = None

    mse_target = torch.sum((pflow-target_flow)**2)
    mse_orig_target = torch.sum((hsflow-target_flow)**2)

    return float(mse.detach().cpu().numpy()), mse_p_gt, float(mse_target.detach().cpu().numpy()), mse_gt_target, float(mse_orig_target.detach().cpu().numpy())

# %% execution


def test_full_attack(delta=.1, optimizer='LBFGS', opt_args={}, seed=None):
    img1_path, img2_path, flow_path = getRandomImagePaths()
    # key='bandage_1', i=30)  # seed=seed)
    # img1_path,img2_path,flow_path='./dataset/mpi_sintel/training/final/temple_3/frame_0027.png', './dataset/mpi_sintel/training/final/temple_3/frame_0028.png', './dataset/mpi_sintel/training/flow/temple_3/frame_0027.flo'
    print(f'analysing: \'{img1_path}\',\'{img2_path}\',\'{flow_path}')

    img1_orig = read_gray_image(img1_path)
    img2_orig = read_gray_image(img2_path)
    im_h, im_w = img1_orig.shape

    # "ground truth"
    flow_hs = torch.tensor(readFloFlow(flow_path)).to(device)
    u_gt, v_gt = flow_hs[:, :, 0], flow_hs[:, :, 1]
    u_gt[u_gt > 1e9] = 0
    v_gt[v_gt > 1e9] = 0

    # plot image and "ground truth"
    show_images(img1_orig, colorplot_light(u_gt, v_gt),
                names=("img2_orig", 'ground truth flow'))

    # %% adverserial attack parameter

    img1 = img1_orig.clone().detach().to(device)
    img2 = img2_orig.clone().detach().to(device)
    img1.requires_grad = True
    img2.requires_grad = True

    _, img1, img2, flow_pert, _, _, _ = full_attack(
        img1_orig, img2_orig, delta_max=delta, flow_hs=flow_hs, balance_factor=1e3, max_iter=1000, target=0, max_iter_hornschunck=1000, optimizer=optimizer, opt_args=opt_args, zero_init=True, pde_type='pde', nebenbed=Relu_globalaverage)

    print(get_metrics(img1, img2, img1_orig,
          img2_orig, flow_pert, flow_hs, None, 0))
    show_images(img1, img2, colorplot_light(flow_pert), save=True)
    return img1, img2, flow_pert


def test_adverserialAttack():
    img1_path, img2_path, flow_path = getRandomImagePaths()
    print(f'analysing: \'{img1_path}\',\'{img2_path}\',\'{flow_path}')

    img1_orig = read_gray_image(img1_path)
    img2_orig = read_gray_image(img2_path)
    im_h, im_w = img1_orig.shape

    # "ground truth"
    from flow_IO import readFloFlow
    flow_gt = torch.tensor(readFloFlow(flow_path)).to(device)
    u_gt, v_gt = flow_gt[:, :, 0], flow_gt[:, :, 1]
    u_gt[u_gt > 1e9] = 0
    v_gt[v_gt > 1e9] = 0

    # plot image and "ground truth"
    show_images(img1_orig, img2_orig, colorplot_light(u_gt, v_gt),
                names=("img1_orig", "img2_orig", 'ground truth flow'))

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
    delta_max = .1

    opt = torch.optim.LBFGS([img1, img2])

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
    img1, img2, loss_hist, found_good_delta = adverserial_attack(
        img1, img2, img1_orig, img2_orig, target[0], target[1], delta_max=delta_max, alpha=alpha, max_iter=max_iter, balance_factor=balance_factor, opt=opt, nebenbed=Relu_globalaverage)
    print('Found good delta: ', found_good_delta)
    # %% show results
    show_images(img1, img2, names=('img1', 'img2'),
                save=True, path='plot.png')
    show_images(img1-img1_orig, img2-img2_orig,
                names=('img1-orig1', 'img2-orig2'))

    # %% get Horn Schunck result from these images
    u, v = horn_schunck_multigrid(
        img1, img2, alpha=alpha, max_iter=max_iter_hornschunck)
    show_images(u, v, colorplot_light(u, v), names=(
        'u', 'v', 'perturbed flow',), colorbars=True)

    show_images(np.array(u)-np.array(u_orig), np.array(v)-np.array(v_orig), colorplot_light(u -
                u_orig, v-v_orig), names=('difference u', 'difference v', 'difference flow'), colorbars=True)

    plt.show()
# test_adverserialAttack()


# %%
if __name__ == "__main__":
    # torch.autograd.anomaly_mode.set_detect_anomaly(True)
    a, b, pflow = test_full_attack(
        delta=.1, optimizer="LBFGS", opt_args={"lr": .1, "max_iter": 10, "history_size": 20})
# %%
