"""Asynchronous epsilon-Thompson Sampling experiment runner

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
    BatchTS
    aegisRandom (~)
    aegisParetoFront (~)
    HalluBatchBO (*)
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
"""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# noqa: E402
import sys  # noqa
import aegis  # noqa
from docopt import docopt  # noqa


if __name__ == "__main__":
    args = docopt(__doc__)

    # get the --args into a nicer form
    epsilon = args["--epsilon"]
    eta = args["--eta"]
    opt_dim = args["--problem_dim"]
    acq_name = args["--acq_name"]
    method = args["<method>"]
    problem_name = args["<problem_name>"]
    n_workers = int(args["<n_workers>"])
    run_no = int(args["<run_no>"])
    budget = int(args["<budget>"])
    time_function = args["<time_function>"]
    repno = args["--repeat_no"]

    # sanity checks for the optional arguments (epsilon and acq_name)
    if ("aegis" in method) and ((epsilon is None) or (eta is None)):
        msg = "When using an aegis acquisition function,"
        msg += " the value of --epsilon and --eta must be set."
        print(msg)
        sys.exit(-1)

    if ("BatchBO" in method) and (acq_name is None):
        msg = "When using a method based on a sequential acquisition"
        msg += " functions, such as EI, UCB or PI, the value of"
        msg += " --acq_name must be set."
        print(msg)
        sys.exit(-1)

    # optional arguments: include the function dimensionality if specified
    problem_params = {"d": int(opt_dim)} if opt_dim is not None else {}

    # get the problem and find out its dimensionality (if not already set)
    if opt_dim is None:
        f_class = getattr(aegis.test_problems, problem_name)
        f = f_class(**problem_params)
        problem_dim = f.dim
    else:
        problem_dim = int(opt_dim)

    # set up the acquisition function parameters
    # same for all
    acq_params = {
        "n_opt_samples": problem_dim * 1000,
        "n_opt_bfgs": 10,
    }
    # only for aegis methods
    if "aegis" in method:
        try:
            epsilon = float(epsilon)
        except ValueError:
            pass
        acq_params["epsilon"] = epsilon
        acq_params["eta"] = float(eta)

    # only for the sequential acquisition function methods
    if "BatchBO" in method:
        acq_params["acq_name"] = acq_name

    # number of Fourier features to build the sample paths with
    if ("aegis" in method) or ("BatchTS" in method):
        acq_params["n_features"] = 2000

    if "aegisExploit" in method:
        acq_params["n_workers"] = n_workers

    # the instance repetition number
    if repno is not None:
        repno = int(repno)

    # deal with the ablation study parameters
    if "ablation" in method:
        acq_params["n_features"] = 2000

        # no workers needed to be specified if no exploitation
        if "NoExploit" not in method:
            acq_params["n_workers"] = n_workers

        # 50/50 split between random and sample path if no exploit
        if "NoExploit" in method:
            acq_params["epsilon"] = 0.5

        # else we use the real parameter 1 / sqrt(d)
        else:
            acq_params["epsilon"] = "sqrtd"

    # finally, perform the optimisation
    aegis.optim.perform_optimisation(
        problem_name=problem_name,
        problem_params=problem_params,
        run_no=run_no,
        budget=budget,
        n_workers=n_workers,
        acq_name=method,
        acq_params=acq_params,
        time_name=time_function,
        repeat_no=repno,
        save_every=10,
    )
