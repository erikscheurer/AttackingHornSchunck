# %%
from os.path import join
from typing import List
from flow_IO import writeFloFlow, readFloFlow
from utilities import *
from adverserial_batched import adverserial_attack
from hornSchunck import horn_schunck_multigrid
from datalogger import Logger
from flow_datasets import getSintelTest, getSintelTrain
import os
import json
from torchvision.io import read_image, ImageReadMode

# %%


def get_univ_pert(img1s, img2s, delta_max, alphas, opt='Adam', max_iter=100, max_iter_hornschunck=None):
    delta = torch.zeros(2, *img1s.shape[1:])
    delta.requires_grad = True

    target_u = torch.zeros_like(img1s)
    target_v = torch.zeros_like(img1s)

    opt = eval(f'torch.optim.{opt}([delta])')

    pertubation, hist = adverserial_attack(img1s, img2s, img1s, img2s, target_u, target_v,
                                           delta_max=delta_max, alphas=alphas, max_iter=max_iter, opt=opt, pertubation=delta)

    if max_iter_hornschunck:
        flows = []
        for img1, img2 in zip(img1s, img2s):
            plt.imshow(img1.cpu()+pertubation[0].detach().cpu())
            plt.imshow(img2.cpu()+pertubation[1].detach().cpu())
            u, v = horn_schunck_multigrid(
                img1+pertubation[0], img2+pertubation[1], alpha=alphas, max_iter=max_iter_hornschunck)
            flows.append([u, v])
        # batch flows back together
        flows = torch.concat(
            [torch.from_numpy(flowFromUV(u, v)).unsqueeze(0) for u, v in flows])
        return pertubation, flows, hist

    # if max_iter_hornschunck is None, dont return flows
    return pertubation, hist
# %%


def process_images(img_paths: List[str], max_imgs: int = None):
    max_imgs = max_imgs or len(img_paths)

    flowLogger = Logger.fromFile('./hornSchunck/logger.json')
    with Logger(subfolder='./results/universal_pertubation2') as logger:
        logger['alphas'] = []
        logger['losses'] = []
        logger['pertubations'] = []

        logger['imgs'] = (img_paths[:max_imgs])
        all_alphas = [flowLogger['alphas'][img]
                      for img in img_paths[:max_imgs-1]]
        logger['alphas'] = all_alphas

        imgs = torch.concat([(read_image(img, ImageReadMode.GRAY)/255).to(device)
                            for img in img_paths[:max_imgs]])
        for n_imgs in range(2, max_imgs):

            img1s = imgs[:n_imgs-1]
            img2s = imgs[1:n_imgs]

            delta_max = .1

            logger['delta_max'] = delta_max

            alphas = all_alphas[:n_imgs-1]
            alphas = torch.tensor(alphas).to(
                device).unsqueeze(1).unsqueeze(1).unsqueeze(1)

            pert, hist = get_univ_pert(
                img1s, img2s, delta_max=delta_max, alphas=alphas, opt='LBFGS')

            logger['losses'].append([l.tolist() for l in hist])
            save_name = join(logger['folder'], str(n_imgs)+'.npy')
            np.save(save_name, pert.detach().cpu().numpy())
            logger['pertubations'].append(save_name)

            logger.toFile()
    return logger['pertubations']


def process_image_pairs(img_pairs: List[List[str]]):

    flowLogger = Logger.fromFile('./hornSchunck/logger.json')
    with Logger(subfolder='./results/universal_pertubation2') as logger:
        logger['alphas'] = []
        logger['losses'] = []
        logger['pertubations'] = []

        logger['imgs'] = (img_pairs)
        all_alphas = [flowLogger['alphas']['/data/erik/' +
                                           '/'.join(img1.split('/')[2:])] for img1, _ in img_pairs]
        logger['alphas'] = all_alphas

        img1s = torch.concat(
            [(read_image(img1, ImageReadMode.GRAY)/255).to(device) for img1, _ in img_paths])
        img2s = torch.concat(
            [(read_image(img2, ImageReadMode.GRAY)/255).to(device) for _, img2 in img_paths])
        for n_imgs in range(2, len(img_pairs)+2):

            img1_current = img1s[:n_imgs-1]
            img2_current = img2s[1:n_imgs]

            delta_max = .1

            logger['delta_max'] = delta_max

            alphas = all_alphas[:n_imgs-1]
            alphas = torch.tensor(alphas).to(
                device).unsqueeze(1).unsqueeze(1).unsqueeze(1)

            pert, hist = get_univ_pert(
                img1_current, img2_current, delta_max=delta_max, alphas=alphas, opt='Adam')

            logger['losses'].append([l.tolist() for l in hist])
            save_name = join(logger['folder'], str(n_imgs)+'.npy')
            np.save(save_name, pert.detach().cpu().numpy())
            logger['pertubations'].append(save_name)

            logger.toFile()
    return logger['pertubations'], logger['jsonname']


def determine_flows(logger_path: str):
    """Im Moment ists nur der flow von den ersten beiden bildern, aber in die pertubation fließen ja immer mehr andere Bilder rein"""
    l = Logger.fromFile(logger_path)
    img1 = (read_image(l.imgs[0][0],
            ImageReadMode.GRAY)/255.).to('cpu').squeeze()
    img2 = (read_image(l.imgs[0][1],
            ImageReadMode.GRAY)/255.).to('cpu').squeeze()
    pflows = []
    for i in range(len(l.pertubations)):
        pertubation = np.load(l.pertubations[i])

        # print(pertubation.max())
        # plt.imshow(img1+pertubation[0])
        # plt.show()
        # plt.imshow(img2+pertubation[1])
        # plt.show()

        pimg1 = img1+pertubation[0]
        pimg2 = img2+pertubation[1]

        u, v = horn_schunck_multigrid(
            pimg1, pimg2, alpha=l.alphas[0], max_iter=1000)

        pflow = flowFromUV(u, v)
        writeFloFlow(pflow, l['pertubations'][i][:-4]+'.flo')
        pflows.append(pflow)

    return pflows


# %%
if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)

    dataset = getSintelTrain('final')
    scene1 = 'alley_1'
    scene2 = 'ambush_2'
    img_paths = batch_list(dataset[scene1]['images'][:2], 2)
    img_paths.extend(batch_list(dataset[scene2]['images'][:2], 2))

    n_imgs = 3
# %%
    pertubation_paths, logger_path = process_image_pairs(img_pairs=img_paths)
    pertubations = [np.load(p) for p in pertubation_paths]

    # pflows = determine_flows('/home/scheurek/adverserial/Adverserial_HornSchunck/results/universal_pertubation2/2022_02_27:17_06_02_1/logger.json')
    pflows = determine_flows(logger_path)

# %%
    path = '/'.join(logger_path.split('/'))
    pflows = [readFloFlow(join(path, f'{i}.flo')) for i in range(2, n_imgs+1)]
    pertubations = [np.load(join(path, f'{i}.npy'))
                    for i in range(2, n_imgs+1)]

# %% evaluation
    flowLogger = Logger.fromFile('./hornSchunck/logger.json')
    flow = readFloFlow(flowLogger['flowResults'][img_paths[0]])
    epes = [avg_EndPointError(pflow, np.zeros_like(pflow)).cpu()
            for pflow in pflows]

# %%
    plt.plot(epes)
    plt.xlabel('n_images')
    plt.ylabel('epe')
    plt.grid(True)
# %% plot some results
    fig, axes = plt.subplots(2, 2)
    [ax.axis('off') for ax in axes.flatten()]

    [axes[0][i].imshow(pertubations[0][i]) for i in [0, 1]]
    [axes[1][i].imshow(pertubations[-1][i]) for i in [0, 1]]
    axes[0, 0].set_title('pert for img1 with 1 imgs')
    axes[0, 1].set_title('pert for img2 with 1 imgs')
    axes[1, 0].set_title('pert for img1 with 40 imgs')
    axes[1, 1].set_title('pert for img2 with 40 imgs')


# %%
