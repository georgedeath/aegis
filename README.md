# Asynchronous ϵ-Greedy Bayesian Optimisation

This repository contains the Python3 code for the experiments and results presented in:

> George De Ath, Richard M. Everson, and Jonathan E. Fieldsend. Asynchronous ϵ-Greedy Bayesian Optimisation. Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, PMLR 161:578-588, 2021. </br>
> **Paper**: https://proceedings.mlr.press/v161/de-ath21a.html

The repository also contains all training data used for the initialisation of
the optimisation runs carried out, the optimisation results of each of the
runs, and jupyter notebooks to generate the results, figures and tables in the
paper.

If you would like help running experiments or just have questions about how the
code works, please open an [issue](https://github.com/georgedeath/aegis/issues)
and we will do our best to offer help and advice.

## Citation

If you use any part of this code in your work, please cite:

```bibtex
@inproceedings{death:aegis:2021,
    title={Asynchronous ϵ-Greedy Bayesian Optimisation},
    author = {George {De Ath} and Richard M. Everson and Jonathan E. Fieldsend},
    year = {2021},
    booktitle = {Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence},
    pages = {578--588},
    year = {2021},
    editor = {de Campos, Cassio and Maathuis, Marloes H.},
    volume = {161},
    series = {Proceedings of Machine Learning Research},
    month = {27--30 Jul},
}
```

## Core packages

- Profet benchmark:
  - `pip install pybnn`
  - `pip install git+https://github.com/amzn/emukit`

- Standard packages:
  - numpy, matplotlib, scipy, statsmodels, torch, gpytorch, botorch, pyDOE2,
    [pygmo2](https://esa.github.io/pygmo2/install.html),
    [pygame](https://www.pygame.org/wiki/GettingStarted),
    [box2d-py](https://pypi.org/project/box2d-py/).

## Reproduction of experiments

The python file `run_exp.py` provides a convenient way to reproduce an
individual experimental evaluation carried out the paper. It has the following
syntax:

```script
> python run_exp.py --help
Usage:
    run_exp.py <problem_name> <run_no> <budget> <n_workers> <method> <time_function>
                [--problem_dim=<d>] [--epsilon=<e>] [--eta=<n>]
                [--acq_name=<acq_name>] [--repeat_no=<repno>]

Arguments:
    <problem_name>  Name of the problem to be optimised.
    <run_no>        Optimisation run number / Problem instance.
    <budget>        Number of function evaluations (including training data)
                    to carry out.
    <n_workers>     Number of asynchronous workers to run.
    <method>        Acquisition function name, choose from one of the list below.
    <time_function> Distribution to sample job times from.
                    Valid options: [halfnorm, pareto, exponential].

Acquisition functions:
    BatchTS [Thompson Sampling]
    aegisRandom (~) [AEGiS-RS]
    aegisParetoFront (~) [AEGiS]
    HalluBatchBO (*) [Kriging Believer]
    LocalPenalisationBatchBO (*) [Local Penalisation]
    HardLocalPenalisationBatchBO (*) [PLAyBOOK]
    (~) = Need to specify epsilon value.
    (*) = Need to specify a sequential acquisition function to use.

Options:
    --problem_dim=<d>   Specify problem dimensionality. Only needed when you're
                        optimising problems with a different dimensionality
                        than the default.
    --epsilon=<e>   Epsilon value for the aegis methods (~).
    --eta=<n>   Eta value for the aegis methods (~).
    --acq_name=<acq_name>   Sequential acquisition function name, only
                            used with methods marked with (*) above.
    --repeat_no=<repno> Used for the instance-based methods. Specifies the
                        run number of the instance. Note that each combination
                        of run_no and repeat_no will need to have its own
                        training data.
```

## Reproduction of figures and tables in the paper

- [AEGiS_results.ipynb](AEGiS_results.ipynb) contains the code to load
and process the optimisation results (stored in `results` directory), as well
as the code to produce all results figures and tables used in the paper and
supplementary material.
- [presentation_plots.ipynb](presentation_plots.ipynb) contains the code to
create the figures used in the presentation at UAI 2021.

## Training data

The initial training locations for each of the 51 sets of
[Latin hypercube](https://www.jstor.org/stable/1268522) samples for the various noise levels are located in the `data` directory. The files are named like `ProblemName_number.pt`, e.g. first set of training locations for the Branin problem is stored in `Branin_001.pt`. Each of these files is a compressed numpy file created with [torch.save](https://pytorch.org/docs/stable/torch.html#torch.save). It has two [torch.tensor](https://pytorch.org/docs/stable/torch.html#torch.tensor) arrays (`Xtr` and `Ytr`) containing the 2*D initial locations and their corresponding fitness values. Note that for problems that have a non-default dimensionality (e.g. Ackley with d=5), then the data files have the dimensionality appended, e.g. `Ackley5_001.pt`; see the suite of [available synthetic test problems](aegis/test_problems/synthetic_problems.py). To load and inspect the training data, use the following instructions:

```python
> python
>>> import torch
>>> data = torch.load('data/Ackley5_001.pt')
>>> Xtr = data['Xtr']  # Training data locations
>>> Ytr = data['Ytr']  # Corresponding function values
>>> Xtr.shape, Ytr.shape
(torch.Size([10, 5]), torch.Size([10]))
```

## Optimisation results

The results of all optimisation runs can be found in the `results` directory.
The notebook [AEGiS_results.ipynb](AEGiS_results.ipynb) shows how to a load all
the experimental data and to subsequently plot it.
