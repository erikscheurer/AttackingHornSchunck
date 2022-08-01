# %%
import argparse
from multiprocessing.sharedctypes import Value
import torch
from adverserial import *
from utilities import *
import flow_IO
import matplotlib.pyplot as plt
from flow_datasets import getSintelTest, getSintelTrain
from datalogger import Logger, evaluate_loggers
import time
from os.path import join
# %%


def initLogger(delta_max, deltaType, optimizer, opt_args, balance_factor, n_examples, max_iter, max_iter_hornschunck, target, zero_init, pde_type, scheduler):
    logger = Logger(subfolder=f'./results/{pde_type}_{target}Target')
    logger["attacktype"] = "L2 error" if delta_max is None else delta_max
    logger["pdeType"] = pde_type
    logger["target"] = target
    logger["delta_max"] = delta_max
    logger["delta_type"] = deltaType
    logger["optimizer"] = optimizer
    logger["optargs"] = opt_args
    logger["scheduler"] = scheduler
    logger["N examples"] = n_examples
    logger["max_iter"] = max_iter
    logger["max_iter_hornschunck"] = max_iter_hornschunck
    logger["alphas"] = []
    logger["n_imgs"] = 0
    logger["img_pairs"] = []
    logger["zero_init"] = zero_init

    logger["balance factor"] = balance_factor

    logger["EPE"] = []  # end point error to original horn schunck result
    logger["EPE GT"] = []  # end point error perturbed to ground truth
    logger["EPE target"] = []  # end point error perturbed to target
    logger["EPE GT target"] = []
    logger["EPE orig target"] = []
    logger["MSE"] = []  # mean squared error to original horn schunck result
    logger["MSE GT"] = []  # mean squared error to ground truth
    logger["MSE target"] = []  # mean squared error to target
    logger["MSE GT target"] = []
    logger["MSE orig target"] = []
    logger["losses"] = []  # loss histories for each image

    logger["deltas"] = {}  # pertubation metrics
    logger["deltas"]["wasfound"] = []
    logger["deltas"]["norm"] = []
    logger["deltas"]["avg"] = []
    logger["deltas"]["abs"] = []
    logger["deltas"]["l2"] = []
    logger["deltas"]["max"] = []
    return logger


def process_dataset(dataset: dict, keys=None, n_examples=None, opt_args={"max_iter": 10, "history_size": 20}, lr='function', scheduler=None, schedule_gamma=.999,
                    target='zero', max_iter=1000, balance_factor=1e8, delta_max=None, optimizer='LBFGS', deltaType="Relu_globalaverage", pde_type='pde', changeOfVar=True,  zero_init=True,  # attack parameter
                    max_iter_hornschunck=1000, alpha_type='optimal',  # horn schunck parameters
                    show=False, save=False, reuseLogger=None, dataset_type='train'):
    """
    Only reuse Logger with the exact same images and deltas! Also the evaluation is then not in order since the values are only appended. Change "append(...)" to "insert(0,...)"
    The evaluation of the other data, EPE, MSe... is useless paired with reusing
    """

    print('\n'*5+f"""
    Processing {dataset_type} dataset with key {keys} and {n_examples} samples.
    Horn Schunck Parameter:          alpha='{alpha_type}' from horn schunck logger
    Adverserial Attack Parameter:   delta={delta_max}, target={target}
    Starting time:                  {time.asctime()}
    """+'\n'*5)
    from utilities import device
    # flowLogger is used to read already determined flows
    flowLogger = Logger.fromFile('./hornSchunck/logger.json')
    if alpha_type == 'optimal':
        save_hs = False
    else:
        alpha = float(alpha_type)

    if lr == 'function':
        if target == 'zero' or target == 'inv':
            lr = .1 * delta_max+0.001
        else:
            lr = .025 * delta_max

    opt_args['lr'] = lr  # insert learning rate into optimizer arguments

    # reuseLogger needs to be tested again
    usedLogger = reuseLogger or initLogger(
        delta_max, deltaType, optimizer, opt_args, balance_factor, n_examples, max_iter, max_iter_hornschunck, target, zero_init, pde_type, scheduler)

    with usedLogger as logger:  # if you leave the context logger will automatically write to file
        logger.toFile()
        try:
            if keys is None:
                keys = dataset.keys()  # if no key is given iterate over *all* keys
                if dataset_type == 'train':  # on training set apply split
                    keys = ["ambush_2", "ambush_6", "bamboo_2",
                            "cave_4", "market_6", "temple_2"]
                    from random import shuffle
                    shuffle(keys)

            for i_key, key in enumerate(keys):
                print(f'\nNow analysing {key}\n')
                # files locations on disk
                flows = dataset[key]["flows"]
                images = dataset[key]["images"]
                if n_examples is None:
                    n_used = len(images)-1
                    indices = list(range(n_used))
                else:
                    indices = np.random.choice(
                        range(len(images)-1), n_used, replace=False)
                # need at least two images for flow
                # images[:n_used+1]
                images1 = getIndexList(images, indices)
                images2 = getIndexList(images, np.array(indices)+1)
                if dataset_type == 'train':
                    if n_examples is not None:
                        # when we have gt, then it has to be the same number as images
                        flows = getIndexList(flows, indices)
                    assert len(flows) == len(
                        images1), f'len(flows)={len(flows)}!=len(images1)={len(images1)}'
                else:
                    flows = [None]*len(images1)
                # iterate over all images in key
                for i_scene, (img1_orig, img2_orig, flow_gt) in enumerate(zip(images1, images2, flows)):
                    # total number of images increases, while counter i_scene is the number of images per scene
                    i = i_key*n_used+i_scene

                    print(f"\nNext image: {img1_orig} \n")

                    if alpha_type == 'optimal':
                        alpha = flowLogger['alphas'][img1_orig]

                    logger['alphas'].append(alpha)

                    logger["img_pairs"].append((img1_orig, img2_orig))

                    # read flows
                    if alpha_type == 'optimal':
                        flow_hs = torch.tensor(flow_IO.readFloFlow(
                            flowLogger["flowResults"][img1_orig])).to(device)
                    else:
                        # if alpha is not optimal we still might have saved it, but now also with alpha as key
                        try:
                            flow_hs = torch.tensor(flow_IO.readFloFlow(
                                flowLogger["flowResults"][img1_orig+str(alpha)])).to(device)
                            save_hs = False
                        except KeyError:
                            flow_hs = None
                            save_hs = True
                            img1_name = img1_orig
                            save_name = key+'_' + \
                                os.path.basename(img1_name)[:-4]+'.flo'

                    if flow_gt:  # is not None: for the testset
                        flow_gt = torch.tensor(
                            flow_IO.readFloFlow(flow_gt)).to(device)

                    # read images
                    img1_orig = read_gray_image(img1_orig)
                    img2_orig = read_gray_image(img2_orig)

                    # execute attack
                    flow_hs, pimg1, pimg2, pflow, target_flow, loss_hist, found_good_delta = full_attack(img1_orig, img2_orig, alpha=alpha, opt_args=opt_args, scheduler=scheduler, schedule_gamma=schedule_gamma,
                                                                                                         target=target, max_iter=max_iter, balance_factor=balance_factor, delta_max=delta_max, optimizer=optimizer, nebenbed=eval(
                                                                                                             deltaType), pde_type=pde_type, changeOfVar=changeOfVar, zero_init=zero_init,  # attack parameter
                                                                                                         max_iter_hornschunck=max_iter_hornschunck, flow_hs=flow_hs,  # horn schunck parameters
                                                                                                         show=show)

                    globalnorm, globalavg, absolute, l2, maximum = get_deltas(
                        pimg1, pimg2, img1_orig, img2_orig)

                    if save_hs:
                        flow_IO.writeFloFlow(flow_hs.cpu().numpy(),
                                             flowLogger.folder + f"/flowResults/alpha_{str(alpha)}_{save_name}")
                        flowLogger["flowResults"][img1_name+str(alpha)] = \
                            flowLogger.folder + \
                            f"/flowResults/alpha_{str(alpha)}_{save_name}"
                        flowLogger.toFile()

                    if save:
                        np.save(join(logger["folder"], f'pertubation_1_{key}_{i:04}'),
                                (pimg1-img1_orig).cpu().detach().numpy())
                        np.save(join(logger["folder"], f'pertubation_2_{key}_{i:04}'),
                                (pimg2-img2_orig).cpu().detach().numpy())
                        # flow_IO.writeFloFlow(flow_hs.cpu().numpy(),
                        #                      join(logger["folder"], f'hs_flow_{key}_{i:04}.flo'))
                        # flow_IO.writeFloFlow(pflow.cpu().numpy(),
                        #                      join(logger["folder"], f'perturbed_flow_{key}_{i:04}.flo'))
                    if show or save:
                        flow_img, fl_max = colorplot_light(
                            flow_hs, return_max=True)
                        show_images(pimg1, pimg2, flow_img, colorplot_light(pflow, auto_scale=False, max_scale=fl_max), names=("perturbed1", 'perturbed2',
                                    "original horn schunck", "perturbed horn schunck"), save=save, show=show, path=join(logger["folder"], f'perturbed_{key}_{i:04}.png'))
                    plt.close()

                    epe, epe_p_gt, epe_target, epe_gt_target, epe_orig_target = get_metrics(
                        pimg1, pimg2, img1_orig, img2_orig, pflow, flow_hs, flow_gt, target_flow)
                    mse, mse_p_gt, mse_target, mse_gt_target, mse_orig_target = get_mse_metrics(
                        pflow, flow_hs, flow_gt, target_flow)

                    logger["EPE"].append(epe)
                    logger["EPE GT"].append(epe_p_gt)
                    logger["EPE target"].append(epe_target)
                    logger["EPE GT target"].append(epe_gt_target)
                    logger["EPE orig target"].append(epe_orig_target)
                    logger["MSE"].append(mse)
                    logger["MSE GT"].append(mse_p_gt)
                    logger["MSE target"].append(mse_target)
                    logger["MSE GT target"].append(mse_gt_target)
                    logger["MSE orig target"].append(mse_orig_target)
                    # append because loss meshes all things together and we want one loss hist per iteration
                    logger["losses"].append(loss_hist)
                    logger["deltas"]["wasfound"].append(found_good_delta)
                    logger["deltas"]["norm"].append(float(globalnorm))
                    logger["deltas"]["avg"].append(float(globalavg))
                    logger["deltas"]["abs"].append(float(absolute))
                    logger["deltas"]["l2"].append(float(l2))
                    logger["deltas"]["max"].append(float(maximum))
                    logger["n_imgs"] += 1
                    logger.toFile()
        except Exception as e:
            print("Optimization failed:", e)

    print('END OF OPTIMIZATION')


# %% evaluate balancefactor loggers


if False:  # __name__ == "__main__":
    path = './results/energy_0Target'
    from datalogger import *
    from utilities import *
    import matplotlib.pyplot as plt

    key = 'EPE target'
    title = 'Adverserial Robustness' if key == 'EPE' else 'Attack Strength'
    evaluate_loggers(path, key=key, deltakey='avg', filter_res=False,
                     useDeltaMax=False, relative='EPE orig target', title=title)
    plt.show()
# %%


def process_function(args):
    # time.sleep(np.random.rand()*10)
    # logging.basicConfig(level=logging.INFO)
    delta = float(args.delta)
    try:
        lr = float(args.lr)
    except ValueError:
        lr = args.lr
    try:
        alpha = float(args.alpha)
    except ValueError:
        alpha = args.alpha
    dev = args.device
    set_device(dev)
    dataset_type = args.dataset
    if dataset_type == 'train':
        dataset = getSintelTrain('final')
    elif dataset_type == 'test':
        dataset = getSintelTest('final')
    target = args.target
    pde_type = args.pdetype
    show = False
    save = str(args.save) == 'True'
    balance_factor = 1.e8
    zero_init = args.zero_init == 'True'
    delta_type = 'Relu_globalaverage'
    n_examples = int(args.n_examples) if isNumber(args.n_examples) else None

    scheduler = 'exponential'
    gamma = .999

    from random import seed
    seed(1)
    np.random.seed(1)

    return process_dataset(dataset, show=show, save=save, keys=None, n_examples=n_examples, target=target, pde_type=pde_type, deltaType=delta_type, delta_max=delta, balance_factor=balance_factor, max_iter=1000, max_iter_hornschunck=20000, optimizer='LBFGS', opt_args={"max_iter": 10, "history_size": 20}, lr=lr, scheduler=scheduler, schedule_gamma=gamma, dataset_type=dataset_type, alpha_type=alpha, zero_init=zero_init)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', default=1e-1)
    parser.add_argument('--lr', default='function')
    parser.add_argument('--device', default='cuda:0',
                        help="determines on which device to run")
    parser.add_argument('--dataset', default='test')
    parser.add_argument('--alpha', default='.1',
                        help="Type of alpha: \nOptions: 'optimal' to look at flowLogger or any float")
    parser.add_argument('--target', default=.1)
    parser.add_argument('--pdetype', default='pde')
    parser.add_argument('--zero_init', default='True')
    parser.add_argument('--n_examples', default=None)
    parser.add_argument('--save', default=True)

    args = parser.parse_args()
    print(args)
    process_function(args)
# %% plot results
    # evaluate_loggers()
    # plt.show()
    # plt.savefig('plot.png')
