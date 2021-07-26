import os
import torch
import aegis

from pyDOE2 import lhs

data_dir = "data"
budget = 200
n_runs = 51
workers = [1, 4, 8, 16]
time_name = "halfnorm"

acq_name = "Random"
acq_params = {}

synth_problems = [
    ("Branin", {}),
    ("Eggholder", {}),
    ("GoldsteinPrice", {}),
    ("SixHumpCamel", {}),
    ("Hartmann3", {}),
    ("Ackley", {"d": 5}),
    ("Hartmann6", {}),
    ("Michalewicz", {"d": 10}),
    ("Rosenbrock", {"d": 10}),
    ("StyblinskiTang", {"d": 10}),
    ("Michalewicz", {"d": 5}),
    ("StyblinskiTang", {"d": 5}),
    ("Rosenbrock", {"d": 7}),
    ("StyblinskiTang", {"d": 7}),
    ("Ackley", {"d": 10}),
]

instance_based_problems = [
    # robot pushing
    ("push4", {}),
    ("push8", {}),
    # profet
    ("svm", {}),
    ("xgboost", {}),
    ("fcnet", {}),
]

# settings for the two problem types
if True:
    problems = synth_problems
    n_repeats = None
else:
    problems = instance_based_problems
    n_repeats = 20

for problem_name, problem_params in problems:
    # get the test problem class
    f_class = getattr(aegis.test_problems, problem_name)

    for run_no in range(1, n_runs + 1):
        for n_workers in workers:
            reprange = (
                [None] if (n_repeats is None) else range(1, n_repeats + 1)
            )

            for repeat_no in reprange:
                # generate the path to the training data file
                filepath = aegis.util.generate_data_filename(
                    problem_name,
                    run_no,
                    problem_params,
                    repeat_no=repeat_no,
                    data_dir=data_dir,
                )

                # load the training data
                data = torch.load(filepath)

                Xtr = data["Xtr"]
                Ytr = data["Ytr"]

                if "problem_params" in data:
                    problem_params.update(data["problem_params"])

                # instantiate the problem rescaled in [0, 1]^d
                f = aegis.util.UniformProblem(f_class(**problem_params))
                f_dim = f.dim

                # samples to generate
                n_train = Xtr.shape[0]
                N_samples = budget - n_train

                # set up the saving paths
                save_path = aegis.util.generate_save_filename(
                    time_name,
                    problem_name,
                    budget,
                    n_workers,
                    acq_name,
                    run_no,
                    problem_params,
                    acq_params,
                    repeat_no=repeat_no,
                )

                if os.path.exists(save_path):
                    print(f"File already exists, skipping: {save_path}")
                    continue

                # Latin hypercube samples
                X = lhs(f_dim, N_samples, criterion="maximin")

                # evaluate
                Y = f(X)

                # combine with initial samples
                _Xtr = torch.zeros((budget, f_dim), dtype=Xtr.dtype)
                _Xtr[:n_train, :] = Xtr
                _Xtr[n_train:, :] = torch.as_tensor(X)

                _Ytr = torch.zeros((budget,), dtype=Ytr.dtype)
                _Ytr[:n_train] = Ytr
                _Ytr[n_train:] = torch.as_tensor(Y)

                # stuff we save, identical to perform_optimisation()
                save_dict = {
                    "problem_name": problem_name,
                    "problem_params": problem_params,
                    "q": 1,
                    "n_workers": n_workers,
                    "acq_name": acq_name,
                    "time_name": time_name,
                    "budget": budget,
                    "Xtr": _Xtr,
                    "Ytr": _Ytr,
                }

                # save
                torch.save(obj=save_dict, f=save_path)
                print(f"Saving: {save_path:s}")
