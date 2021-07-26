import os
import torch
import numpy as np

from pyDOE2 import lhs
from . import test_problems, util


def generate_training_data_LHS(
    problem_name,
    n_exp_start=1,
    n_exp_end=51,
    n_samples=None,
    n_repeats=None,
    additional_arguments={},
):
    exp_nos = np.arange(n_exp_start, n_exp_end + 1)
    N = len(exp_nos)

    # check that there are the same number of arguments as there are
    # experimental training data to construct
    for _, v in additional_arguments.items():
        assert len(v) == N, (
            "There should be as many elements for each "
            "optional arguments as there are experimental runs"
        )

    # get the function class
    f_class = getattr(test_problems, problem_name)

    for i, run_no in enumerate(exp_nos):
        # get the optional arguments for this problem instance (if they exist)
        problem_params = {k: v[i] for (k, v) in additional_arguments.items()}

        # instantiate the function, uniform wrap it and torch wrap it
        f = util.TorchProblem(util.UniformProblem(f_class(**problem_params)))

        # default data type
        dtype = f.yopt.dtype

        # if n_samples isn't specified, generate 2 * D samples
        n_samples = 2 * f.dim if (n_samples is None) else n_samples

        # if we've got no repeats, then just set up a loop with one item,
        # which corresponds to setting repeat_no=None for one run.
        reprange = [None] if (n_repeats is None) else range(1, n_repeats + 1)

        for repeat_no in reprange:

            save_path = util.generate_data_filename(
                problem_name, run_no, problem_params, repeat_no=repeat_no
            )

            if os.path.exists(save_path):
                print(f"File exists, skipping: {save_path:s}")
                continue

            # storage
            D = {"problem_params": problem_params}

            # LHS
            D["Xtr"] = torch.as_tensor(
                lhs(f.dim, n_samples, criterion="maximin"), dtype=dtype
            )
            D["Ytr"] = f(D["Xtr"])

            # save the training data
            torch.save(obj=D, f=save_path)
            print("Saved: {:s}".format(save_path))


def generate_push4_targets(N):
    """Generates target locations for the 'push4' problems by LHS.
    Generates ``N`` target locations, defined as [x, y] pairs each in [-5, 5],
    using Latin hypercube sampling.
    Parameters
    ----------
    N : int
        Number of locations to generate.
    Returns
    -------
    T1_x : (N, ) numpy.ndarray
        x-axis location of the targets.
    T1_y : (N, ) numpy.ndarray
        y-axis locations of the targets.
    """
    # lower and upper bounds
    T_lb = np.array([-5, -5])
    T_ub = np.array([5, 5])

    # LHS sample and rescale from [0, 1]^2 to the bounds above
    T = lhs(2, N, criterion="maximin") * (T_ub - T_lb) + T_lb

    T1_x, T1_y = T.T
    return T1_x, T1_y


def generate_push8_targets(N):
    """Generates pairs of target locations for the two robots in push8.
    The function generates two set of ``N`` LHS samples and pairs up the
    locations such that the distance between them is >= 1.1, thereby giving
    room for the objects (with diameter = 1) to each sit perfectly on their
    target without blocking one another.
    Parameters
    ----------
    N : int
        Number of pairs of locations to generate.
    Returns
    -------
    T1_x : (N, ) numpy.ndarray
        x-axis location of the first targets in the pairs
    T1_y : (N, ) numpy.ndarray
        y-axis locations of the first targets in the pairs
    T2_x : (N, ) numpy.ndarray
        x-axis location of the second targets in the pairs
    T2_y : (N, ) numpy.ndarray
        y-axis locations of the second targets in the pairs
    """
    # we can call generate_push4_targets(N) twice to get two sets of locations
    T1 = np.concatenate([generate_push4_targets(N)]).T
    T2 = np.concatenate([generate_push4_targets(N)]).T

    # ensure the pair of targets is greater than 1.1 away from each other by
    # shuffling the samples to be paired together - thus not allowing objects
    # to overlap with each other if they are perfectly positioned on the target
    while True:
        norm = np.linalg.norm([T1 - T2], axis=1)
        if np.min(norm) >= 1.1:
            break

        np.random.shuffle(T2)

    T1_x, T1_y = T1.T
    T2_x, T2_y = T2.T

    return T1_x, T1_y, T2_x, T2_y


def generate_profet_training_data(n_repeats=20):
    names = ["svm", "xgboost", "fcnet"]

    problem_params = {"problem_instance": np.arange(1, 52)}

    for name in names:
        generate_training_data_LHS(
            name, additional_arguments=problem_params, n_repeats=n_repeats
        )


def generate_push_training_data(n_repeats=20):
    # generate per-instance target locations
    T1_x, T1_y = generate_push4_targets(51)
    push4_targets = {"t1_x": T1_x, "t1_y": T1_y}

    # generate the training data
    generate_training_data_LHS(
        "push4", additional_arguments=push4_targets, n_repeats=n_repeats
    )

    # generate per-instance target locations
    T1_x, T1_y, T2_x, T2_y = generate_push8_targets(51)
    push8_targets = {"t1_x": T1_x, "t1_y": T1_y, "t2_x": T2_x, "t2_y": T2_y}

    # generate the training data
    generate_training_data_LHS(
        "push8", additional_arguments=push8_targets, n_repeats=n_repeats
    )


def generate_synthetic_training_data():
    synth_problems = [
        # #### initial problems
        # "Branin"
        # "Eggholder",
        # "GoldsteinPrice",
        # "SixHumpCamel",
        # "Hartmann3",
        # "Ackley:5",
        # "Hartmann6",
        # "Michalewicz:10",
        # "Rosenbrock:10",
        # "StyblinskiTang:10",
        # #### further problems
        "Michalewicz:5",
        "StyblinskiTang:5",
        "Rosenbrock:7",
        "StyblinskiTang:7",
        "Ackley:10",
    ]

    for problem_name in synth_problems:
        additional_arguments = {}

        if ":" in problem_name:
            problem_name, dim = problem_name.split(":")
            dim = int(dim)
            additional_arguments["d"] = [dim] * 51

        generate_training_data_LHS(
            problem_name,
            additional_arguments=additional_arguments,
            n_repeats=None,
        )


if __name__ == "__main__":
    # n_repeats = 20
    # generate_profet_training_data(n_repeats=n_repeats)
    # generate_push_training_data(n_repeats=n_repeats)

    generate_synthetic_training_data()
