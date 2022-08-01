# %%
from __future__ import with_statement
import json
import logging
import time
import os
from os.path import join, sep
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utilities import ignore, itercount


def get_full_folder(subfolder='./results'):
    creationtime = time.localtime()
    return subfolder+sep+f'{creationtime[0]}_'+f'{creationtime[1]:02}_'+f'{creationtime[2]:02}:'+f'{creationtime[3]:02}_'+f'{creationtime[4]:02}_'+f'{creationtime[1]:02}'


def set_default(obj):
    if isinstance(obj, set):
        logging.debug('Switching set to list')
        return list(obj)
    raise TypeError


def is_jsonable(x):
    try:
        json.dumps(x, default=set_default)
        return True
    except:
        return False

# %%


class Logger(object):
    def __init__(self, full_name=None, subfolder='./results', name='logger', full_folder=None, serializable=True, reuse=False, **kwargs) -> None:
        """Create logger. If there is no file/folder in the given location, then it will create them

        Keyword Arguments:
            full_name {str} -- Full path to .pkl file. Overwrites every other entry (default: {None})
            subfolder {str} -- folder to save results in (default: {'./results'})
            name {str} -- Name of the .pkl file (default: {'logger'})
            full_folder {str} -- full folder including time date (default: {None})
            serializable {bool} -- pass if the Logger can be written to json or else be pickled
            reuse {bool} -- if loaded from file and reuse=True it will write again. Enables adding to previously done loggers.
            **kwargs {dict} -- All other information will be saved in logger dictionary.
        """
        self.dict = kwargs
        # if dict is empty, logger comes from File, don't rewrite times
        self.loaded = bool(kwargs)
        self.reuse = reuse
        self.serializable = serializable

        if self.loaded:
            return  # if the logger is already loaded, the following things are already set

        if full_name is None:  # construct a new name
            # get folder
            if full_folder is None:
                folder = get_full_folder(subfolder=subfolder)
            else:
                folder = full_folder

            # np.random.randint(1,10000) # if the path exists already, construct a new folder
            i = 1
            while os.path.exists(folder+f'_{i}'):
                i = i+1  # np.random.randint(1,1000)
            os.makedirs(folder+f'_{i}')
            folder = folder+f'_{i}'

            # set full file name
            filename = name if name.endswith('.pkl') else name+'.pkl'
            full_name = join(folder, filename)
        else:  # create folders
            name = os.path.basename(full_name).split('.')[0]
            if not os.path.exists(os.path.dirname(full_name)):
                os.makedirs(os.path.dirname(full_name))

        self.dict["fullname"] = full_name
        self.dict["jsonname"] = join(os.path.dirname(full_name), name+'.json')
        self.dict["folder"] = os.path.dirname(full_name)
        self.dict["starttime"] = time.asctime()
        self.writeToFile()

    @classmethod
    def reuseLogger(cls, logger_path):
        return Logger.fromFile(logger_path, reuse=True)

    @classmethod
    def fromFile(cls, fullname=None, fullfolder='', name='logger.json', reuse=False):
        """reloads logger from given file and will write to the same file.
        give either fullname or fullfolder and name (with file ending)

        Keyword Arguments:
            fullname {str} -- Full path to pkl file (default: {None})
            fullfolder {str} -- full folder including time (default: {''})
            name {str} -- Name of the pkl file (default: {'logger'})

        Returns:
            Logger -- The reloaded logger
        """
        if fullname is not None:
            fullname = fullname
        else:
            fullname = join(fullfolder, name)
        with open(fullname, 'rb') as inputFile:
            if fullname.endswith('.json'):
                prev = json.load(inputFile)
            else:
                prev = pickle.load(inputFile)

        # pickle only saves the dictionary
        return cls(prev['fullname'], reuse=reuse, **prev)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.loaded or self.reuse:
            self['endtime'] = time.asctime()
            self.toFile()

    def __getitem__(self, key):
        return self.dict[key]

    def __getattribute__(self, __name: str):
        try:
            return object.__getattribute__(self, __name)
        except AttributeError as e:
            with ignore(KeyError):
                return self[__name]
            raise e

    def __setitem__(self, key, value) -> None:
        self.dict[key] = value

    def __str__(self) -> str:
        return f'Logger object at {self["fullname"]}:\n{self.dict}'

    def keys(self):
        return self.dict.keys()

    def values(self):
        return self.dict.values()

    def writeToFile(self) -> None:
        """writes current state of the logger to the disk.
        """
        if self.serializable:
            try:
                json.dump(self.dict, open(
                    self["jsonname"], 'w'), indent=4, default=set_default)
            except TypeError:
                print(
                    '\nError: The Logger is not serializable!\nChanging mode to pickle')
                print("List of keys not serializable:", [
                      key for key in self.keys() if not is_jsonable(self[key])])
                self.serializable = False
                self.writeToFile()
                os.remove(self['jsonname'])
        else:
            with open(join(self["fullname"]), 'wb') as writefile:
                pickle.dump(self.dict, writefile, pickle.HIGHEST_PROTOCOL)
    toFile = writeToFile

# %%


def get_logger_list(folder='./results'):
    executions = os.listdir(folder)
    logger_list = []

    # extract all executions from the given folders
    for execution in filter(lambda x: os.path.isdir(join(folder, x)), executions):
        files = os.listdir(join(folder, execution))
        # filter out the logger-files
        loggerfiles = [x for x in files if (
            x.endswith('.pkl') or x.endswith('.json'))]
        assert not len(
            loggerfiles) > 1, f'more than one logger file in directory {folder}/{execution}'
        if len(loggerfiles) == 1:
            logger_list.append(join(folder, execution, loggerfiles[0]))
    return sorted(logger_list)


def load_latest_logger(folder='./results'):
    logger_file = get_logger_list(folder)[-1]
    l = Logger.fromFile(logger_file)
    return l

# %%


def eval_convergence(loggers, ax=None, inTitle=lambda logger: '', delta_key='avg'):
    if ax is None:
        fig, ax = plt.subplots()
    # overview where the runs were sucessfull and where they weren't
    for l in loggers:
        nancount = (itercount(l['EPE'], np.isnan))
        failed_count = (
            itercount(l['deltas'][delta_key], lambda x: x > l["delta_max"]))
        ax.bar(0, len(l['deltas'][delta_key])-failed_count -
               nancount, .2, label='sucessfull', color='limegreen')
        ax.bar(.2, failed_count, .2, label='failed', color='tab:red')
        ax.bar(-.2, nancount, .2, label='NaN', color='black')
        ax.invert_xaxis()
        ax.legend(loc='upper left')
        ax.set_title(
            "$\delta_{max}$ = "+f"{l['delta_max']:.1e} and {inTitle(l)}")


def evaluate_loggers(folder='./results', loggers=[], filter_res=False, key="EPE", deltakey='avg', useDeltaMax=True, relative=False, title=None, label='', ylabel=None, xlabel='$\|\delta_1,\delta_2\|$', style='.-', xscale='log', yscale='log', **kwargs):
    """Iterates over all loggers in the given folder and plots the given key.

    Keyword Arguments:
        folder {str} -- folder in which to search for loggers (default: {'./results'})
        loggers {list} -- optional list of loggers. Overwrites the folders argument. (default: {[]})
        filter_res {bool} -- if True filters only for perturbations where the constraint was fulfilled (default: {False})
        key {str} -- keyword under which the metric is found in the logger (default: {"EPE"})
        deltakey {str} -- which delta type to plot against. (default: {'avg'})
        useDeltaMax {bool} -- If True uses the given constraint as x-axis instead of the average perturbation (default: {True})
        relative {str} -- the key to divide the values of the logger by to get the relative result. eg for key='EPE target' use relative='EPE orig target' (default: {False})
        title {str} -- title of the resulting plot (default: None). If None, no title is added.
        label {str} -- label of the plotted curve (default: {''})

    Returns:
        _type_ -- _description_
    """
    if not loggers:
        loggers = get_logger_list(folder)
        loggers = [Logger.fromFile(l) for l in loggers]
    loggers.sort(key=lambda l: l["delta_max"])
    deltas = []
    epes = []  # end point errors

    for l in loggers:
        if filter_res:

            def filter_index(i: int) -> bool:
                """checks if at index the delta of the logger fulfilles the criteria.
                To sort out the images where the optimisation didn't reach delta_max

                Keyword Arguments:
                    goal {float} -- the goal, l['deltas'] should be smaller than (default: {(l["delta_max"])})
                """
                try:
                    goal = l['delta_max']
                except KeyError as e:
                    print('KeyError', e, 'using nanmedian(deltas) as goal')
                    goal = np.nanmedian(l["deltas"])
                tol = goal*10  # tolerance that delta can be larger than the goal
                if l['deltas'][deltakey][i] == l['deltas'][deltakey][i] and l['deltas'][deltakey][i]-goal < tol:
                    return True
                return False

            if key in 'losses':
                values = [min(losses)
                          for losses in np.array(l['losses'])[:, 1]]
            else:
                values = l[key]
                if relative:
                    rel_values = np.array(l[relative])
                    values = np.array(values)/rel_values
            filtered = [e for i, e in enumerate(values) if filter_index(i)]
            if not filtered:  # if the list of usable values is empty
                print(f'list of filtered {key} is empty')
                return

            epes.append(np.nanmean(filtered))
            if useDeltaMax:
                deltas.append(l["delta_max"])
            else:  # use actual delta value as x value
                deltas.append(np.mean([delta for i, delta in enumerate(
                    l['deltas'][deltakey]) if filter_index(i)]))

        else:  # unfiltered results
            if useDeltaMax:
                deltas.append(l['delta_max'])
            else:
                deltas.append(np.nanmean(l["deltas"][deltakey]))
            if key in 'losses':
                values = [min(losses)
                          for losses in np.array(l['losses'])[:, 1]]
            else:
                values = l[key]
                if relative:
                    rel_values = np.array(l[relative])
                    values = np.array(values)/rel_values

            epes.append(np.nanmean(values))
    print(f'deltas: {deltas}')
    print(f'epes: {epes}')
    plt.plot(deltas, epes, style, label=label, **kwargs)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.grid(True)
    plt.xlabel(xlabel)
    ylabel = ylabel or key
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)


def evaluate_lr_loggers(folder, **kwargs):
    """Iterates over all loggers and plots the mean End point error of one logger (one dataset iteration) over the mean pertubation size

    Keyword Arguments:
        folder {str} -- Folder where all loggers are stored (default: {'./results'})
    """
    loggers = get_logger_list(folder)
    loggers = [Logger.fromFile(l) for l in loggers]
    lrs = list(set([l['optargs']['lr'] for l in loggers]))
    lrs.sort()

    print('eval convergence')
    fig, axes = plt.subplots(1, len(lrs))
    for lr, ax in zip(lrs, axes):
        lr_loggers = list(filter(lambda l: l['optargs']['lr'] == lr, loggers))
        eval_convergence(lr_loggers, ax=ax, inTitle=lambda l: f'lr={lr:.2e}')
    plt.show()
    print("now evaluation")
    plt.figure()
    for lr in lrs:
        lr_loggers = list(filter(lambda l: l['optargs']['lr'] == lr, loggers))
        modified_kwargs = kwargs.copy()
        modified_kwargs['label'] = kwargs['label'] + f', lr={lr:.2e}'
        evaluate_loggers(loggers=lr_loggers, **modified_kwargs)
    plt.legend()


if __name__ == "__main__":
    print("evaluate_loggers is now only compatible with a logger where 'delta' is a container for all deltas")
    evaluate_loggers(folder='./results/balancefactor',
                     deltakey='avg', key="EPE", filter_res=True)

# %% results from current run
    evaluate_loggers(folder='./results/secondRun',
                     filter_res=True, key="EPE", useDeltaMax=True)
    loggers = [Logger.fromFile(l)
               for l in get_logger_list('./results/secondRun/')]

# %% results from first run
    loggers = [Logger.fromFile(l) for l in get_logger_list(
        './results/firstFullRun/')]  # %%
    evaluate_loggers(folder='./results/firstFullRun/',
                     filter_res=True, key="EPE", useDeltaMax=False)
# %% this is one case in first run where delta is nan
    print(loggers[1]['deltas'][88])
    print(loggers[1]['img_pairs'][88])
    plt.plot(loggers[1]['losses'][88][-100:], '.-')
# %% here delta is not nan but larger than deltamax
    print(loggers[1]['deltas'][-1])
    print(loggers[1]['img_pairs'][-1])
    plt.plot(loggers[1]['losses'][-1][100:], '.-')
# %% normal loss
    print(loggers[0]['deltas'][2])
    print(loggers[0]['img_pairs'][2])
    plt.plot(loggers[2]['losses'][-1], '.-')
    plt.show()
    plt.plot(loggers[2]['losses'][-1][-100:], '.-')

# %%
