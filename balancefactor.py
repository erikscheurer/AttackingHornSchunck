# %%
import argparse
import torch
from adverserial import *
from utilities import *
import flow_IO
import matplotlib.pyplot as plt
from flow_datasets import getSintelTrain
from datalogger import Logger
import time
# %%


def initLogger(delta_max, deltaType, optimizer, opt_args, n_examples, max_iter, max_iter_hornschunck, target):
    logger = Logger(subfolder='./results/balancefactor')
    logger["attacktype"] = "L2 error" if delta_max is None else delta_max
    logger["target"] = target
    logger["delta_max"] = delta_max
    logger["delta_type"] = deltaType
    logger["optimizer"] = optimizer
    logger["optargs"] = opt_args
    logger["N examples"] = n_examples
    logger["max_iter"] = max_iter
    logger["max_iter_hornschunck"] = max_iter_hornschunck
    logger["alphas"] = []
    logger["n_imgs"] = 0
    logger["img_pairs"] = []

    logger["balance factors"] = set()
    logger["optbalance"] = []
    logger["allEPEs"] = []
    logger["allhistlen"] = []
    logger["alldeltas"] = []
    # for each balancefactor if delta<delta_max was found
    logger["allcontraintfulfilled"] = []

    logger["EPE"] = []  # end point error to original horn schunck result
    logger["EPE GT"] = []  # end point error perturbed to ground truth
    logger["EPE target"] = []  # end point error perturbed to target
    logger["MSE"] = []  # mean squared error to original horn schunck result
    logger["MSE GT"] = []  # mean squared error to ground truth
    logger["MSE target"] = []  # mean squared error to target
    logger["losses"] = []  # loss histories for each image

    logger["deltas"] = {}  # pertubation metrics
    logger["deltas"]["wasfound"] = []
    logger["deltas"]["sum"] = []
    logger["deltas"]["avg"] = []
    logger["deltas"]["abs"] = []
    logger["deltas"]["l2"] = []
    logger["deltas"]["max"] = []
    return logger


def process_dataset(dataset: dict, key='alley_1', n_examples=None, opt_args={"max_iter": 10, "history_size": 20}, lr='function',
                    target='zero', max_iter=1000, balance_factor_min=1e8, balance_factor_max=1e8, delta_max=None, optimizer='LBFGS', deltaType="Relu_globalaverage", changeOfVar=True,  # attack parameter
                    max_iter_hornschunck=1000,  # horn schunck parameters
                    show=False, reuseLogger=None):
    """
    Only reuse Logger with the exact same images and deltas! Also the evaluation is then not in order since the values are only appended. Change "append(...)" to "insert(0,...)"
    The evaluation of the other data, EPE, MSe... is useless paired with reusing
    """

    print('\n'*5+f"""
    Processing dataset with key {key} and {n_examples} samples.
    Horn Schunck Parameter:          alpha='optimal' from horn schunck logger
    Adverserial Attack Parameter:   delta={delta_max}, target={target}
    balance factor range: ({balance_factor_min,balance_factor_max}) in steps of *10
    Starting time:                  {time.asctime()}
    """+'\n'*5)
    from utilities import device
    # flowLogger is used to read already determined flows
    flowLogger = Logger.fromFile('./hornSchunck/logger.json')

    if lr == 'function':
        lr = 0.99*delta_max+0.001  # minimum is 0.001
    opt_args['lr'] = lr

    usedLogger = reuseLogger or initLogger(
        delta_max, deltaType, optimizer, opt_args, n_examples, max_iter, max_iter_hornschunck, target)
    with usedLogger as logger:  # if you leave the context logger will automatically write to file
        logger.toFile()
        try:
            if key is None:
                keys = dataset.keys()  # if no key is given iterate over *all* keys
                keys = ["ambush_2", "ambush_6", "bamboo_2",
                        "cave_4", "market_6", "temple_2"]
            else:
                keys = [key]

            for i_key, key in enumerate(keys):
                print(f'\nNow analysing {key}\n')
                # files locations on disk
                flows = dataset[key]["flows"]
                images = dataset[key]["images"]
                if n_examples is not None:
                    flows = flows[:n_examples]
                    # need at least two images for flow
                    images = images[:n_examples+1]

                assert len(flows) == len(
                    images[:-1]), f'len(flows)={len(flows)}!=len(images)={len(images)-1}'

                # iterate over all images in key
                for i_scene, (img1_orig, img2_orig, flow_gt) in enumerate(zip(images[:-1], images[1:], flows)):
                    # total number of images increases, while counter i_scene is the number of images per scene
                    i = i_key*n_examples+i_scene

                    print(f"\nNext image: {img1_orig} \n")
                    alpha = flowLogger['alphas'][img1_orig]
                    logger['alphas'].append(alpha)

                    logger["img_pairs"].append((img1_orig, img2_orig))

                    # read flows
                    flow_hs = torch.tensor(flow_IO.readFloFlow(
                        flowLogger["flowResults"][img1_orig])).to(device)
                    flow_gt = torch.tensor(
                        flow_IO.readFloFlow(flow_gt)).to(device)

                    # read images
                    img1_orig = read_gray_image(img1_orig)
                    img2_orig = read_gray_image(img2_orig)

                    # execute attack
                    best_balance = balance_factor_max
                    best_epe_target = 1e15
                    best_metrics = []
                    # this is the global balance factor, found_good_delta will be the bool value if the constraint is fulfilled in each run
                    # if so far the best balancefactor didnt lead to a small enough delta, then also save balances where the deltamax wasnt reached
                    balance_found_good_delta = False

                    if reuseLogger is None:  # if we reuse a old logger, all images were already processed
                        logger["alldeltas"].append([])
                        logger["allcontraintfulfilled"].append([])
                        logger["allEPEs"].append([])
                        logger["allhistlen"].append([])
                        assert len(logger["allEPEs"]) == i + \
                            1, "length doesn't work"
                    balance_factor = balance_factor_max*10
                    while balance_factor/10 >= balance_factor_min:
                        balance_factor /= 10
                        print('now with balance_factor', balance_factor)
                        try:
                            logger["balance factors"].add(balance_factor)
                        except AttributeError:  # if logger is reused, balancefactor would be a list instead of a set
                            logger["balance factors"] = set(
                                logger["balance factors"])
                            logger["balance factors"].add(balance_factor)
                        flow_hs, pimg1, pimg2, pflow, target_flow, loss_hist, found_good_delta = full_attack(img1_orig, img2_orig, alpha=alpha, opt_args=opt_args,
                                                                                                             target=target, max_iter=max_iter, balance_factor=balance_factor, delta_max=delta_max, optimizer=optimizer, nebenbed=eval(
                                                                                                                 deltaType), changeOfVar=changeOfVar,  # attack parameter
                                                                                                             max_iter_hornschunck=max_iter_hornschunck, flow_hs=flow_hs,  # horn schunck parameters
                                                                                                             show=show)

                        globalsum, globalavg, absolute, l2, maximum = get_deltas(
                            pimg1, pimg2, img1_orig, img2_orig)

                        if not np.isnan(loss_hist[0][-1]):

                            # show_images(pimg1, pimg2, colorplot_light(flow_hs), colorplot_light(pflow), names=("perturbed1", 'perturbed2', "original horn schunck", "perturbed horn schunck"), save=True, path=join(logger["folder"], f'result_{key}_{i:04}.png'))
                            # plt.close()

                            epe, epe_p_gt, epe_target = get_metrics(
                                pimg1, pimg2, img1_orig, img2_orig, pflow, flow_hs, flow_gt, target_flow)
                            mse, mse_p_gt, mse_target = get_mse_metrics(
                                pflow, flow_hs, flow_gt, target_flow)
                            logger["allEPEs"][i].append(float(epe))
                            logger["allhistlen"][i].append(len(loss_hist[0]))
                            logger["allcontraintfulfilled"][i].append(
                                found_good_delta)
                            logger["alldeltas"][i].append(float(globalavg))

                            # If the constraint was either not met by any balance_factor OR this balance factor does fulfill it, THEN Compare if current epe is the best and THEN save balance factor as the best.
                            # target is more important than epe to horn schunck
                            if (not balance_found_good_delta or found_good_delta) and epe_target < best_epe_target:

                                if not balance_found_good_delta and found_good_delta:
                                    # if the constraint was fulfilled for the first time with this balance_factor, then update
                                    balance_found_good_delta = True

                                best_balance = balance_factor
                                best_epe_target = epe_target
                                best_metrics = [epe, epe_p_gt, epe_target, mse, mse_p_gt,
                                                mse_target, loss_hist, globalsum, globalavg, absolute, l2, maximum]

                        else:  # np.isnan(loss_hist[-1])
                            logger["allEPEs"][i].append(np.nan)
                            logger["allcontraintfulfilled"][i].append(
                                found_good_delta)
                            logger["allhistlen"][i].append(np.nan)

                    assert logger["allEPEs"][i], "'allEPEs' is empty!"
                    logger["optbalance"].append(best_balance)

                    logger["EPE"].append(float(best_metrics[0]))
                    logger["EPE GT"].append(float(best_metrics[1]))
                    logger["EPE target"].append(float(best_metrics[2]))
                    logger["MSE"].append(float(best_metrics[3]))
                    logger["MSE GT"].append(float(best_metrics[4]))
                    logger["MSE target"].append(float(best_metrics[5]))
                    # append because loss meshes all things together and we want one loss hist per iteration
                    logger["losses"].append(best_metrics[6])
                    logger["deltas"]["wasfound"].append(found_good_delta)
                    logger["deltas"]["sum"].append(float(best_metrics[7]))
                    logger["deltas"]["avg"].append(float(best_metrics[8]))
                    logger["deltas"]["abs"].append(float(best_metrics[9]))
                    logger["deltas"]["l2"].append(float(best_metrics[10]))
                    logger["deltas"]["max"].append(float(best_metrics[11]))
                    logger["n_imgs"] += 1
                    logger.toFile()
        except Exception as e:
            print("Optimization failed:", e)

    print('END OF OPTIMIZATION')


def eval_convergence(loggers, inTitle=lambda logger: ''):

    # overview where the runs were sucessfull and where they weren't
    for l in loggers:
        factors = np.sort(l['balance factors'])[::-1]
        epes = np.array(l["allEPEs"])
        deltas = np.array(l["alldeltas"])
        nancount = []
        failed_count = []
        for i in range(len(factors)):  # in number of balance factors
            nancount.append(itercount(epes[:, i], np.isnan))
            failed_count.append(
                itercount(deltas[:, i], lambda x: x > l["delta_max"]))
        plt.bar(np.arange(len(factors)), epes.shape[0]-np.array(nancount)-np.array(
            failed_count), .2, label='sucessfull', color='limegreen')
        plt.bar(np.arange(len(factors))+.2, failed_count, .2,
                label='failed', color='tab:red')
        plt.bar(np.arange(len(factors))-.2,
                nancount, .2, label='NaN', color='black')
        plt.xticks(np.arange(len(factors)), [
                   '$10^'+str({np.log10(b).astype(int)}) + '$' for b in factors])
        plt.gca().invert_xaxis()
        plt.xlabel('balance factor')
        plt.legend(loc='upper left')
        plt.title(
            "Sucessfull vs Unsucessfull vs NaN runs for $\delta_{max}$ = "+f"{l['delta_max']:.1e} and {inTitle(l)}")
        plt.show()

# %% evaluate balancefactor loggers


def eval_different_balancefactors(path='./results/balancefactor', loggers=None) -> None:
    """makes some plots for different balancefactors but with one fix learning rate. Evaluate a folder path or given loggers

    Keyword Arguments:
        path {str} -- folder path where the logggers are stored (default: {'./results/balancefactor'})
        loggers {List(loggers)} -- List of loggers to evaluate, optional. 
    """
    if loggers is None:
        from datalogger import get_logger_list
        loggers = [Logger.fromFile(l) for l in get_logger_list(path)]

    opt = []
    delta_max = []
    std = []
    for l in loggers:
        factors = np.sort(l['balance factors'])[::-1]
        opt.append(np.mean(l['optbalance']))
        std.append(np.std(l['optbalance']))
        delta_max.append(l['delta_max'])

        # overview for all optimal balances for one delta_max
        plt.plot(factors, [l['optbalance'].count(f) for f in factors], '.-')
        plt.title('optbalance overview for $\delta_{max}=$'+str(delta_max[-1]))
        plt.ylabel('count of imgs where factor was optimal')
        plt.xlabel('factors')
        plt.xscale('log')
        plt.show()

    # overview for mean opt-balances (not that meaningful as deltas are logarithmic -> mean not great)
    # plt.plot(delta_max,opt,'.-')
    plt.errorbar(delta_max, opt, yerr=std, fmt='.')
    plt.xscale('log')
    plt.xlabel("delta_max")
    plt.ylabel("mean optimal balance")
    plt.show()

    eval_convergence(loggers)

    # overview in which scenes the convergence failed
    def getscene(imgpath):
        return imgpath.split('/')[-2]

    for l in loggers[:]:
        factors = np.sort(l['balance factors'])[::-1]
        deltas = np.array(l["alldeltas"])
        fails = {getscene(img): 0 for img, _ in l['img_pairs']}
        for i, factor in enumerate(factors):
            for j, delta in enumerate(deltas[:, i]):
                if delta > l["delta_max"]:
                    fails[getscene(l['img_pairs'][j][0])] += 1
                    # print(f'failed with {delta},{factor}:',l['img_pairs'][j][0].split('/')[-2:])
        plt.bar(range(len(fails)), fails.values())
        plt.xticks(range(len(fails)), fails.keys())
        plt.title(
            f'failed imgs in scenes with delta {l["delta_max"]}. All balance factors')
        plt.show()
    # overview after how many iterations the algorithm converged
    for l in loggers:
        factors = np.sort(l['balance factors'])[::-1]
        histlen = np.array(l["allhistlen"])
        meanhist = np.nanmean(histlen, axis=0)
        plt.plot(factors, meanhist, '.-')
        plt.xscale('log')
        plt.xlabel('balance factor')
        plt.ylabel('number of iterations to convergence')
        plt.title("$\delta_{max}$ = "+str(l['delta_max']))
        plt.show()
# %%


if __name__ == "__main__":
    path = './results/balancefactor_function_lr'  # _different_LR'
    import matplotlib.pyplot as plt
    import numpy as np
    from datalogger import Logger, get_logger_list
    loggers = [Logger.fromFile(l) for l in get_logger_list(path)]
    deltas = [l['delta_max'] for l in loggers]
    lrs = [l['optargs']['lr'] for l in loggers]
    key = 'EPE target'

    # for lr in set(lrs):
    #     lr_loggers = [loggers[i] for i, rate in enumerate(lrs) if rate == lr]
    #     lr_loggers.sort(key=lambda l: l['delta_max'])
    #     # eval_different_balancefactors(loggers=lr_loggers)
    #     epes = [l[key] for l in lr_loggers]

    #     for i, epe_list in enumerate(epes):
    #         plt.plot(epe_list, '.', label=lr_loggers[i]['delta_max'])
    #     plt.legend()
    #     plt.title(f'different deltas for {len(epes[0])} images with lr={lr}')
    #     plt.xlabel('image number')
    #     plt.ylabel(key)
    #     plt.show()

    for delta in set(deltas):
        delta_loggers = [loggers[i]for i, d in enumerate(deltas) if d == delta]
        delta_loggers.sort(key=lambda l: l['optargs']['lr'])

        delta_loggers = delta_loggers

        # eval_convergence(
        #     delta_loggers, inTitle=lambda logger: f'lr = {logger["optargs"]["lr"]}')

        epes = [l[key] for l in delta_loggers]

        for i, epe_list in enumerate(epes):
            plt.plot(epe_list, '.', label=delta_loggers[i]['optargs']['lr'])
        plt.legend()
        plt.title(
            f'different learning rates for {len(epes[0])} images with delta={delta}')
        plt.xlabel('image number')
        plt.ylabel(key)

        plt.show()
        plt.close()

        plt.plot([0]*len(epes[0]), '.',
                 label=delta_loggers[0]['optargs']['lr'])
        for i, epe_list in enumerate(epes[1:]):
            plt.plot([epe - ref_epe for epe, ref_epe in zip(epe_list,
                     epes[0])], '.', label=delta_loggers[i+1]['optargs']['lr'])
        plt.legend()
        plt.title(
            f'differences in {key} for {len(epes[0])} images with delta={delta}')
        plt.xlabel('image number')
        plt.ylabel(key)

        plt.show()
        plt.close()

        # print the actual minimum
        # loss is either [img1:[[loss],[pdeloss],[nebenloss],img2:...]
        mincount_loss = [0]*len(delta_loggers)
        mincount_epe = [0]*len(delta_loggers)
        mincount_epe_target = [0]*len(delta_loggers)
        for i in range(len(delta_loggers[0]['img_pairs'])):
            lossPerDelta = [min(l['losses'][i][0]) if len(l['losses'][0]) == 3
                            else min(l["losses"][i]) for l in delta_loggers]
            minindex = np.argmin(lossPerDelta)
            mincount_loss[minindex] += 1
            # print(f'lr with minimum loss is {delta_loggers[minindex]["optargs"]["lr"]}')
            targetEPEPerDelta = [l['EPE target'][i] for l in delta_loggers]
            minindex = np.argmin(targetEPEPerDelta)
            mincount_epe_target[minindex] += 1
            # print(f'lr with minimum EPE target is {delta_loggers[minindex]["optargs"]["lr"]}')
            EPEPerDelta = [l['EPE'][i] for l in delta_loggers]
            minindex = np.argmin(EPEPerDelta)
            mincount_epe[minindex] += 1
            # print(f'lr with minimum EPE is {delta_loggers[minindex]["optargs"]["lr"]}')
        print(f'minimum for loss: {mincount_loss}')
        print(f'minimum for epe: {mincount_epe}')
        print(f'minimum for epe_target: {mincount_epe_target}')
        print(
            f'where indexes are in this order: {[l["optargs"]["lr"] for l in  delta_loggers]}')

        for logger in delta_loggers:
            # if is for legacy logging where loss is not saved seperately
            if len(logger["losses"][0]) == 3:
                [plt.plot(losses[0]) for losses in logger['losses']]
                plt.title(
                    f'loss for delta {delta}, lr {logger["optargs"]["lr"]}')
                plt.yscale('log')
                plt.show()
                # [plt.plot(losses[1]) for losses in logger['losses']]
                # plt.title(
                #     f'pde-loss for delta {delta}, lr {logger["optargs"]["lr"]}')
                # plt.yscale('log')
                # plt.show()
                # [plt.plot(losses[2]) for losses in logger['losses']]
                # plt.title(
                #     f'neben-loss for delta {delta}, lr {logger["optargs"]["lr"]}')
                # plt.yscale('log')
                # plt.show()
            else:
                [plt.plot(losses) for losses in logger['losses']]
                plt.title(
                    f'loss for delta {delta}, lr {logger["optargs"]["lr"]}')
                plt.yscale('log')
                plt.show()
            plt.close()
        # if __name__ == "__main__":
        #     eval_different_lrs()
    plt.close()


# %%
def process_function(args):
    # time.sleep(np.random.rand()*10)
    # logging.basicConfig(level=logging.INFO)
    delta = float(args.delta)
    try:
        lr = float(args.lr)
    except ValueError:
        lr = args.lr
    dev = args.device
    set_device(dev)
    dataset = getSintelTrain('final')

    return process_dataset(dataset, key=None, n_examples=10, delta_max=delta, max_iter=1000, max_iter_hornschunck=1000, optimizer='LBFGS', opt_args={"max_iter": 10, "history_size": 20}, lr=lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', default=.01)
    parser.add_argument('--lr', default='function')
    parser.add_argument('--device', default='cuda:0',
                        help="determines on which device to run")

    args = parser.parse_args()

    process_function(args)
# %% plot results
    # evaluate_loggers()
    # plt.show()
    # plt.savefig('plot.png')
# %%
