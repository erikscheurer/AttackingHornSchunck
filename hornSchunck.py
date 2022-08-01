# %% imports
from os.path import join
from time import time
from typing import Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from datalogger import Logger
from flow_datasets import getSintelTest, getSintelTrain
from utilities import *
import scipy.sparse as sp
from functools import lru_cache

# %% define helper function


@lru_cache
def get_sparse_neigbour_matrix_and_count(im_h, im_w, torch=False):
    """Constructs the Matrix that can be multiplied with u_k or v_k to result in the sum of the neigbours

    Args:
        im_h (int): Height of the resulting image
        im_w (int): Width of the resulting image

    Returns:
        scipy.sparse.coo_matrix: Matrix shape=(im_w*im_h,im_w*im_h)
        np.ndarray: Vector number of Neigbours for each pixel. Shape=(im_w*im_h)
    Or:
        torch.
    """
    data = []
    rows = []
    cols = []
    count = np.zeros(im_h*im_w)
    for i in range(im_h*im_w):  # iterate over image
        row = i // im_w
        col = i % im_w

        up = i-im_w
        down = i+im_w
        left = i-1
        right = i+1

        if col != 0:  # check left
            count[i] += 1
            data.append(1)
            rows.append(i)
            cols.append(left)
        if col != im_w-1:  # check right
            count[i] += 1
            data.append(1)
            rows.append(i)
            cols.append(right)
        if row != 0:  # check up
            count[i] += 1
            data.append(1)
            rows.append(i)
            cols.append(up)
        if row != im_h-1:  # check down
            count[i] += 1
            data.append(1)
            rows.append(i)
            cols.append(down)
    if torch:
        from utilities import device
        tensor = torch.sparse_coo_tensor(
            indices=[cols, rows], values=data, size=(im_h*im_w, im_h*im_w)).float()
        return tensor, torch.from_numpy(count).float().to(device)
    matrix = sp.coo_matrix((data, (rows, cols)))
    return matrix, count


@convertTypes('torch')
def horn_schunck_torch(img1, img2, u0=None, v0=None, alpha=1, max_iter=100, plot=False, log=True):
    """solves horn Schunck Model with Jacobi solver. This version is entirely in torch and can thereby collect gradients.
    The Torch version is much slower than normal hornSchunck, probably because torch cant do sparse matrix multiplication with non-sparse matrices
    and I have to create sparse matrices sparse_v and sparse_u, that aren't actually sparse (-> more memory and probably copying takes time?)

    Args:
        img1 (torch.tensor)
        img2 (torch.tensor)
        alpha (int, optional): controlls smoothing. Defaults to 1.
        max_iter (int, optional): Number of maximum iterations for the Jacobi solver. Defaults to 100.
        plot (bool, optional): If True, will plot every 10 iterations and save in folder
    Returns:
        u,v: flow in x and y direction. Shape the same as img1
    """
    # calculate helping variables
    im_h, im_w = img1.shape
    img_x = get_f_x(img1).ravel()
    img_y = get_f_y(img1).ravel()
    img_z = get_f_z(img1, img2).ravel()

    u_k = u0.ravel() if u0 is not None else torch.rand_like(img_x)
    v_k = v0.ravel() if v0 is not None else torch.rand_like(img_x)

    neighbour_matrix, neighbour_count = get_sparse_neigbour_matrix_and_count(
        im_h, im_w, torch=True)
    sp_indices = torch.cat([torch.arange(0, u_k.shape[0], dtype=int).unsqueeze(
        0), torch.zeros_like(u_k, dtype=int).unsqueeze(0)])  # indices for "sparse" vector
    for k in range(1, max_iter+1):

        # plot image each iteration
        if plot and k % 10 == 1:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            p1 = ax1.imshow(u_k.reshape((im_h, im_w)), cmap='gray')
            plt.colorbar(p1, ax=ax1)
            p2 = ax2.imshow(v_k.reshape((im_h, im_w)), cmap='gray')
            plt.colorbar(p2, ax=ax2)
            # plt.savefig(f'./hornSchunck_iterations/iter_{k:04}')
            plt.show()
            # plt.pause(.1)

        u_k1 = u_k.clone()  # previous iteration
        v_k1 = v_k.clone()  # previous iteration

        sparse_u_k = torch.sparse_coo_tensor(indices=sp_indices, values=u_k1)
        sparse_v_k = torch.sparse_coo_tensor(indices=sp_indices, values=v_k1)
        # update image
        u_k = (-img_x*(img_y*v_k1 + img_z) + alpha*torch._sparse_sparse_matmul(neighbour_matrix,
               sparse_u_k).to_dense().squeeze())/(img_x**2+alpha*neighbour_count)
        v_k = (-img_y*(img_x*u_k1 + img_z) + alpha*torch._sparse_sparse_matmul(neighbour_matrix,
               sparse_v_k).to_dense().squeeze())/(img_y**2+alpha*neighbour_count)

        if printing_enabled and log and k % 50 == 0:
            changeu = torch.norm(u_k-u_k1)
            changev = torch.norm(v_k-v_k1)
            if changev+changeu < 1e-10:
                print(' Skipping iterations')
                return u_k.reshape((im_h, im_w)), v_k.reshape((im_h, im_w))
            progress_bar(k, max_iter, title='HornSchunck',
                         msg=f'change to before: u:{changeu:.05}, v:{changev:.05}')
    return u_k.reshape((im_h, im_w)), v_k.reshape((im_h, im_w))


# @convertTypes('numpy')
def horn_schunck(img1, img2, u0=None, v0=None, alpha=.1, max_iter=100, plot=False, log=True):
    """solves horn Schunck Model with Jacobi solver

    Args:
        img1 (np.ndarray)
        img2 (np.ndarray)
        alpha (int, optional): controlls smoothing. Defaults to 1.
        max_iter (int, optional): Number of maximum iterations for the Jacobi solver. Defaults to 100.
        plot (bool, optional): If True, will plot every 10 iterations and save in folder
    Returns:
        u,v: flow in x and y direction. Shape the same as img1
    """
    # calculate helping variables
    im_h, im_w = img1.shape
    img_x = get_f_x(img1, asnumpy=True).ravel()
    img_y = get_f_y(img1, asnumpy=True).ravel()
    img_z = get_f_z(img1, img2).ravel()

    u_k = u0.ravel() if u0 is not None else np.random.rand(*img_x.shape)
    v_k = v0.ravel() if v0 is not None else np.random.rand(*img_x.shape)

    neighbour_matrix, neighbour_count = get_sparse_neigbour_matrix_and_count(
        im_h, im_w)

    for k in range(1, max_iter+1):

        # plot image each iteration
        if plot and k % 10 == 1:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            p1 = ax1.imshow(u_k.reshape((im_h, im_w)), cmap='gray')
            plt.colorbar(p1, ax=ax1)
            p2 = ax2.imshow(v_k.reshape((im_h, im_w)), cmap='gray')
            plt.colorbar(p2, ax=ax2)
            # plt.savefig(f'./hornSchunck_iterations/iter_{k:04}')
            plt.show()
            # plt.pause(.1)

        u_k1 = u_k.copy()  # previous iteration
        v_k1 = v_k.copy()  # previous iteration

        # update image
        u_k = (-img_x*(img_y*v_k1 + img_z) + alpha *
               neighbour_matrix.dot(u_k1))/(img_x**2+alpha*neighbour_count)
        v_k = (-img_y*(img_x*u_k1 + img_z) + alpha *
               neighbour_matrix.dot(v_k1))/(img_y**2+alpha*neighbour_count)

        if printing_enabled and log and k % 50 == 0:
            changeu = np.max(abs(u_k-u_k1))
            changev = np.max(abs(v_k-v_k1))
            if changev+changeu < 1e-10:
                print(' Skipping iterations')
                return u_k.reshape((im_h, im_w)), v_k.reshape((im_h, im_w))
            progress_bar(k, max_iter, title='HornSchunck',
                         msg=f'change to before: u:{changeu:.05}, v:{changev:.05}')
    return u_k.reshape((im_h, im_w)), v_k.reshape((im_h, im_w))


def get_hornschunck_problem(fx, fy, fz, imh, imw, alpha):
    """
    Returns the Matrix and right hand side for the Horn-Schunck algorithm
    fx,fy,fz must be raveled
    """
    from scipy.sparse import dia_matrix, coo_matrix
    data = (np.concatenate((fx*fx, fy*fy)),
            np.pad(fx*fy, (0, imh*imw), 'constant'),
            np.pad(fx*fy, (imh*imw, 0), 'constant'))
    offsets = [0, -imh*imw, imh*imw]
    pde_matrix = dia_matrix((data, offsets), shape=(2*imh*imw, 2*imh*imw))

    neighbour_matrix, n_count = get_sparse_neigbour_matrix_and_count(imh, imw)
    data = np.concatenate((neighbour_matrix.data, neighbour_matrix.data))
    rows = np.concatenate((neighbour_matrix.row, imh*imw+neighbour_matrix.row))
    cols = np.concatenate((neighbour_matrix.col, imh*imw+neighbour_matrix.col))

    neighbour_matrix = coo_matrix((data, (rows, cols)),
                                  shape=(2*imh*imw, 2*imh*imw)) - \
        dia_matrix((np.concatenate((n_count, n_count)), [0]),
                   shape=(2*imh*imw, 2*imh*imw))

    rhs = -np.concatenate((fx*fz, fy*fz))

    return pde_matrix-alpha*neighbour_matrix, rhs


@convertTypes('numpy')
def horn_schunck_nonJacobi(img1, img2, u0=None, v0=None, alpha=.1, max_iter=1000, tol=1e-2):
    """solves horn Schunck Model with BiCGStab

    Args:
        img1 (np.ndarray)
        img2 (np.ndarray)
        alpha (int, optional): controlls smoothing. Defaults to 1.
        max_iter (int, optional): Number of maximum iterations for the Jacobi solver. Defaults to 100.
        plot (bool, optional): If True, will plot every 10 iterations and save in folder
    Returns:
        u,v: flow in x and y direction. Shape the same as img1
    """
    # calculate helping variables
    im_h, im_w = img1.shape
    img_x = get_f_x(img1, asnumpy=True).ravel()
    img_y = get_f_y(img1, asnumpy=True).ravel()
    img_z = get_f_z(img1, img2).ravel()

    from scipy.sparse.linalg import spsolve, bicgstab
    A, b = get_hornschunck_problem(img_x, img_y, img_z, im_h, im_w, alpha)

    if u0 is not None:
        x0 = np.concatenate((u0.ravel(), v0.ravel()))
    else:
        x0 = None
    uv, info = bicgstab(A, b, x0=x0, maxiter=max_iter)
    if info == 0:
        print('converged')
    else:
        print('not converged')
    u, v = uv[:len(uv)//2], uv[len(uv)//2:]
    u = u.reshape(img1.shape)
    v = v.reshape(img1.shape)

    return u, v, info


@convertTypes('numpy')
def horn_schunck_multigrid_nonJacobi(img1, img2, alpha=1, max_iter=100, levels=5, plot=False) -> Tuple[np.ndarray, np.ndarray]:
    im_h, im_w = img1.shape

    from PIL import Image
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)

    uk = np.ones((im_h//2**(levels+1), im_w//2**(levels+1)))
    vk = np.ones((im_h//2**(levels+1), im_w//2**(levels+1)))

    if printing_enabled:
        print(f'Horn Schunck for {levels} stages. Current level...')
    for level in range(levels, -1, -1):
        if printing_enabled:
            print(f'{levels-level}...', end='')
        uk = np.array(Image.fromarray(uk).resize(
            (im_w//2**level, im_h//2**level), Image.ANTIALIAS))
        vk = np.array(Image.fromarray(vk).resize(
            (im_w//2**level, im_h//2**level), Image.ANTIALIAS))

        img1_lowres = img1.resize(
            (im_w//2**level, im_h//2**level), Image.ANTIALIAS)
        img2_lowres = img2.resize(
            (im_w//2**level, im_h//2**level), Image.ANTIALIAS)
        img1_lowres = np.array(img1_lowres)
        img2_lowres = np.array(img2_lowres)

        uk, vk, _ = horn_schunck_nonJacobi(img1_lowres, img2_lowres, u0=uk, v0=vk,
                                           alpha=alpha, max_iter=max_iter)

        # from flow_plot import colorplot_light
        # show_images(colorplot_light(flowFromUV(uk, vk)))
    return uk, vk


@convertTypes('numpy')
def horn_schunck_multigrid(img1, img2, alpha=1, max_iter=100, levels=5, plot=False) -> Tuple[np.ndarray, np.ndarray]:
    im_h, im_w = img1.shape

    from PIL import Image
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)

    uk = np.ones((im_h//2**(levels+1), im_w//2**(levels+1)))
    vk = np.ones((im_h//2**(levels+1), im_w//2**(levels+1)))

    if printing_enabled:
        print(f'Horn Schunck for {levels} stages. Current level...', end='')
    for level in range(levels, -1, -1):
        if printing_enabled:
            print(f'{levels-level}...', end='')
        uk = np.array(Image.fromarray(uk).resize(
            (im_w//2**level, im_h//2**level), Image.ANTIALIAS))
        vk = np.array(Image.fromarray(vk).resize(
            (im_w//2**level, im_h//2**level), Image.ANTIALIAS))

        img1_lowres = img1.resize(
            (im_w//2**level, im_h//2**level), Image.ANTIALIAS)
        img2_lowres = img2.resize(
            (im_w//2**level, im_h//2**level), Image.ANTIALIAS)
        img1_lowres = np.array(img1_lowres)
        img2_lowres = np.array(img2_lowres)
        log = (level == 0)  # only log in the last level
        if log and printing_enabled:
            print()

        uk, vk = horn_schunck(img1_lowres, img2_lowres, u0=uk, v0=vk,
                              alpha=alpha, max_iter=max_iter, plot=plot, log=log)

    return uk, vk


def horn_schunck_withPDEloss(img1, img2, u0=None, v0=None, alpha=1, max_iter=100, plot=False, loss_every=50):
    from utilities import device
    from adverserial import PDE_loss
    im_h, im_w = img1.shape

    u_k = u0 or np.random.rand(im_h*im_w)
    v_k = u0 or np.random.rand(im_h*im_w)

    for k in range(max_iter//loss_every):
        u_k, v_k = horn_schunck(img1, img2, u0=u_k.ravel(
        ), v0=v_k.ravel(), alpha=alpha, max_iter=loss_every, plot=plot)
        if printing_enabled:
            print(f'PDE loss after {(k+1)*loss_every} iterations:', PDE_loss(torch.tensor(img1), torch.tensor(img2), torch.tensor(u_k, dtype=torch.float32).to(device), torch.tensor(
                v_k, dtype=torch.float32).to(device), get_lapl_f(torch.tensor(u_k, dtype=torch.float32).to(device)), get_lapl_f(torch.tensor(v_k, dtype=torch.float32).to(device)), alpha).cpu().numpy())
        u_k, v_k = np.array(u_k), np.array(v_k)
    return u_k, v_k


@convertTypes('torch')
def horn_schunck_PDEtorch(img1, img2, u0=None, v0=None, alpha=1, max_iter=100, plot=False):
    from utilities import device
    if u0 is None:
        u = torch.zeros_like(img1).to(device)
        v = torch.zeros_like(img1).to(device)
    else:
        u = torch.tensor(u0).to(device)
        v = torch.tensor(v0).to(device)
    u.requires_grad = True
    v.requires_grad = True

    from adverserial import PDE_loss

    opt = torch.optim.Adam([u, v])

    def closure():
        opt.zero_grad()

        laplu = get_lapl_f(u)
        laplv = get_lapl_f(v)

        loss = PDE_loss(img1, img2, u, v, laplu, laplv, alpha)

        loss.backward()
        return loss

    for it in range(1, max_iter+1):
        l = closure()
        opt.step(closure)
        progress_bar(it, max_iter, title="HornSchunck torch",
                     msg=f'loss = {l}')

    return u, v


@convertTypes('torch')
def horn_schunck_EnergyTorch(img1, img2, u0=None, v0=None, alpha=1, max_iter=100, plot=False):
    from utilities import device
    if u0 is None:
        u = torch.zeros_like(img1).to(device)
        v = torch.zeros_like(img1).to(device)
    else:
        u = torch.tensor(u0).to(device)
        v = torch.tensor(v0).to(device)

    u.requires_grad = True
    v.requires_grad = True

    img_x = get_f_x(img1).squeeze().to(device)
    img_y = get_f_y(img1).squeeze().to(device)
    img_z = get_f_z(img1, img2).squeeze().to(device)

    def energy(u, v):
        u_x = get_f_x(u).squeeze()
        u_y = get_f_y(u).squeeze()
        v_x = get_f_x(v).squeeze()
        v_y = get_f_y(v).squeeze()

        return torch.sum((img_x*u+img_y*v+img_z)**2+alpha*(u_x**2+u_y**2+v_x**2+v_y**2))

    opt = torch.optim.LBFGS([u, v])  # , lr=.1, max_iter=10, history_size=20)

    print("initial energy:", energy(u, v))

    def closure():
        opt.zero_grad()

        loss = energy(u, v)

        loss.backward()
        return loss

    for it in range(1, max_iter+1):
        l = closure()
        opt.step(closure)
        progress_bar(it, max_iter, title="HornSchunck Energy torch",
                     msg=f'loss = {l}')

    return u, v


@convertTypes('numpy')
def find_alpha(img1, img2, flow_gt, alpharange, maxiter=200):
    global printing_enabled
    printing_enabled = False

    def loss(alpha):
        alpha = np.array(alpha)
        u, v = horn_schunck_multigrid(
            img1, img2, alpha=alpha, max_iter=maxiter)
        flow = np.concatenate(
            (np.expand_dims(u, -1), np.expand_dims(v, -1)), axis=-1)
        return avg_EndPointError(flow, flow_gt).cpu().numpy()
    from scipy.optimize import minimize, minimize_scalar
    # res = minimize(loss,x0=np.array(alpharange).mean(),tol=1e-4,options={'maxiter':15, 'disp':True})
    res = minimize_scalar(loss, bounds=alpharange, method='Bounded')
    print(f'Optimization converged to {res["x"]}')
    printing_enabled = True
    return res['x']


def get_optimal_alpha_dataset():
    logging.basicConfig(level=logging.INFO)
    logging.debug('debug')
    logging.info('info')

    import json
    import os
    from flow_IO import writeFloFlow

    with open('config.json') as config:
        data = json.load(config)
    os.environ["DATASETS"] = data["dataset_path"]

    dataset = getSintelTest('final')  # getSintelTrain('final')

    keys = dataset.keys()
    alpha_results = {}
    flow_results = {}
    with Logger(full_name='./hornSchunck_test/flow_logger.pkl') as logger:

        for scene in keys:
            for i in range(len(dataset[scene]['images'])-1):
                print('Analysing', scene, 'image', i)
                # Read images (in black and white)
                img1_name = dataset[scene]['images'][i]
                img1 = img1_name
                img2 = dataset[scene]['images'][i+1]
                img1 = plt.imread(img1).sum(-1)
                img2 = plt.imread(img2).sum(-1)
                im_h, im_w = img1.shape

                # "ground truth"
                from flow_IO import readFloFlow
                flow_gt = readFloFlow(dataset[scene]['flows'][i])

                alpha = find_alpha(img1, img2, flow_gt, (1e-5, 5))
                alpha_results[img1_name] = float(alpha)
                print(
                    '\n'*3+f'For {scene} Number {i} alpha={alpha} seems to be optimal'+'\n'*3)

                optimal_hornschunck = horn_schunck_multigrid(
                    img1, img2, alpha=alpha, max_iter=500)
                save_name = join(
                    logger['folder'], scene+'_'+os.path.basename(img1_name)[:-4]+'.flo')
                writeFloFlow(optimal_hornschunck, save_name)
                flow_results[img1_name] = save_name

                # save results
                with open('alphas.json', 'w') as alpha_resultfile:
                    print(alpha_results)
                    json.dump(alpha_results, alpha_resultfile)
                logger['alphas'] = alpha_results
                logger['flowResults'] = flow_results
                logger.toFile()


# %%
if __name__ == "__main__":
    from flow_IO import readFloFlow
    from flow_datasets import getSintelTrain
    from flow_plot import *
    from utilities import *
    dataset = getSintelTrain('final')
    scene = "bandage_1"
    i = 30
    img1, img2, flow_gt = getRandomImagePaths(key=scene, i=i)
    img1 = read_gray_image(img1).cpu().numpy()
    img2 = read_gray_image(img2).cpu().numpy()
    im_h, im_w = img1.shape
    flow_gt = readFloFlow(flow_gt)

    # %%
    u, v = horn_schunck_EnergyTorch(
        img1, img2, u0=u, v0=v, max_iter=100, alpha=.1)
    show_images(colorplot_light(flowFromUV(u, v)))

    # %%
    u, v = horn_schunck_multigrid_nonJacobi(
        img1, img2, max_iter=100, alpha=.1, levels=5)

    show_images(colorplot_light(flowFromUV(u, v)))
    print(avg_EndPointError(flow_gt, flowFromUV(u, v)))

    # %%
    u, v, _ = horn_schunck_nonJacobi(
        img1, img2, max_iter=1000, alpha=.1)

    # %%
    show_images(colorplot_light(flowFromUV(u, v)))
    print(avg_EndPointError(flow_gt, flowFromUV(u, v)))

    # %%
    start = time()
    u, v = horn_schunck(img1, img2, max_iter=1000)
    show_images(colorplot_light(flowFromUV(u, v)))
    print(time()-start)
    print(avg_EndPointError(flow_gt, flowFromUV(u, v)))

    # %%
    start = time()
    u_m, v_m = horn_schunck_multigrid(img1, img2, max_iter=1000)
    show_images(colorplot_light(flowFromUV(u, v)))
    print(avg_EndPointError(flow_gt, flowFromUV(u_m, v_m)))
    print(time()-start)
    # img1 = torch.from_numpy(img1).requires_grad_(True).to(device)
    # start = time()
    # print(horn_schunck_torch(img1, img2, max_iter=100))
    # print(time()-start)

    # %%
    show_images(colorplot_light(flow_gt))
