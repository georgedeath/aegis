import os
import torch
import tqdm
import numpy as np
from scipy import stats
from joblib import Parallel, delayed

import aegis


def compute_ranks(
    errors, start_idx=0, pool=None, n_jobs=-1, n_bootstrap=1000
) -> np.ndarray:
    """
    Computes the averaged ranking score in every iteration and for every task..
    :param errors: matrix with M x I x R x N entries, where M are the number
    of optimizers, I are the number of instances or tasks, R is the number of
    runs per task and N are the number of function evaluations per task and run
    :param n_bootstrap: number bootstrap samples to compute the ranks
    :return: the ranks after each iteration
    """
    n_methods = errors.shape[0]
    n_instances = errors.shape[1]
    n_runs = errors.shape[2]
    n_iters = errors.shape[3]

    ranks = np.zeros([n_methods, n_iters])

    rr = np.arange(n_methods)

    # precompute all run indices
    run_inds = np.random.randint(
        0, n_runs, size=(n_instances, n_bootstrap, n_methods)
    )

    def rd(bootstrap_no):
        runs = run_inds[instance_id, bootstrap_no, :]
        instance_runs = errors[rr, instance_id, runs, start_idx:]
        rank_samples = stats.rankdata(instance_runs, method="average", axis=0)
        return rank_samples

    if pool is None:
        pool = Parallel(n_jobs=n_jobs, backend="threading")

    with tqdm.tqdm(total=n_instances, leave=False) as pbar:
        for instance_id in range(n_instances):
            rs = pool(
                delayed(rd)(bootstrap_no)
                for bootstrap_no in range(n_bootstrap)
            )

            for r in rs:
                ranks[:, start_idx:] += r
            pbar.update()

    ranks /= 1 * (n_instances * n_bootstrap)

    return ranks


def create_savefile_name(
    data_dir, time_func, n_workers, problem_name, pres=False
):
    save_name_list = [
        "rankdata",
        f"_{time_func}",
        f"_{n_workers}",
        f"_{problem_name}",
        "_pres" if pres else "",
        ".npz",
    ]

    return os.path.join(data_dir, "".join(save_name_list))


# ---- settings
data_dir = r"results"

for_presentation = True

paper_names = {
    "Random": "Random",
    "BatchTS": "TS",
    "HalluBatchBO-EI": "KB (EI)",
    "LocalPenalisationBatchBO-EI": "LP (EI)",
    "HardLocalPenalisationBatchBO-EI": "PLaYBOOK (EI)",
    "aegisExploitRandom-sqrtd": "eTSE-RS (1/sqrtd)",
    "aegisExploitParetoFront-sqrtd": "eTSE-PF (1/sqrtd)",
}

if for_presentation:
    del paper_names["aegisExploitRandom-sqrtd"]

method_names = [key for key in paper_names]

print("Running on methods:", method_names)

workers = [4, 8, 16]

instance_problems = [
    ("push4", {}),
    ("push8", {}),
    ("svm", {}),
    ("fcnet", {}),
    ("xgboost", {}),
]

time_functions = [
    "halfnorm",
    # "pareto"
    # "exponential"
]

budget = 200
start_run = 1
end_run = 51
n_repeats = 20  # 1 to 21

# number of boostrap samples
n_bootstrap = 1000

# --------------------------------------------------------------------------- #
# ---- load the data
# --------------------------------------------------------------------------- #
D = {}

total = (
    len(instance_problems)
    * len(method_names)
    * (end_run - start_run + 1)
    * n_repeats
    * len(workers)
    * len(time_functions)
)
print("Loading the data:")
with tqdm.tqdm(total=total, leave=True) as pbar:
    for time_func in time_functions:
        D[time_func] = {}

        for n_workers in workers:
            D[time_func][n_workers] = {}

            for problem_name, problem_params in instance_problems:
                save_path = create_savefile_name(
                    data_dir,
                    time_func,
                    n_workers,
                    problem_name,
                    for_presentation,
                )

                if os.path.exists(save_path):
                    pbar.write(f"Loading data: Path exists: {save_path}")

                    pbar.update(
                        (end_run - start_run + 1)
                        * len(method_names)
                        * n_repeats
                    )

                    continue

                res = np.zeros(
                    (
                        len(method_names),  # M
                        end_run - start_run + 1,  # I
                        n_repeats,  # R
                        budget,  # N
                    )
                )

                f_class = getattr(aegis.test_problems, problem_name)
                f = f_class(**problem_params)

                for m, method_name in enumerate(method_names):
                    acq_params = {}

                    if "-" in method_name:
                        mn, eps_or_acq, *rest = method_name.split("-")

                        # only for aegis methods
                        if "aegis" in method_name:
                            if not isinstance(eps_or_acq, str):
                                eps_or_acq = float(eps_or_acq)
                            acq_params["epsilon"] = eps_or_acq

                        elif "BatchBO" in method_name:
                            acq_params["acq_name"] = eps_or_acq

                        else:
                            err = f"Invalid method name: {method_name:s}"
                            raise ValueError(err)

                    else:
                        mn = method_name

                    for i, run_no in enumerate(range(start_run, end_run + 1)):
                        for r, repeat_no in enumerate(range(1, n_repeats + 1)):
                            fn = aegis.util.generate_save_filename(
                                time_func,
                                problem_name,
                                budget,
                                n_workers,
                                mn,
                                run_no,
                                problem_params,
                                acq_params,
                                repeat_no=repeat_no,
                            )

                            try:

                                data = torch.load(fn)
                                Ytr = data["Ytr"].numpy().ravel()
                                n = Ytr.size

                                res[m, i, r, :n] = Ytr

                                if n != budget:
                                    print("Not full:", fn, Ytr.shape)

                            except FileNotFoundError:
                                print("Missing", os.path.basename(fn))
                                # raise
                            except:
                                print(method_name)
                                print(mn)
                                print(fn)
                                raise

                            pbar.update()

                res = np.abs(res - f.yopt.ravel()[0])
                res = np.minimum.accumulate(res, axis=-1)

                D[time_func][n_workers][problem_name] = res

# --------------------------------------------------------------------------- #
# ---- compute the ranking data for each problem + n_workers combination
# --------------------------------------------------------------------------- #
total = len(instance_problems) * len(time_functions) * len(workers)

with tqdm.tqdm(total=total) as pbar, Parallel(
    n_jobs=-1, backend="loky"
) as pool:
    for problem_name, problem_params in instance_problems:
        for time_func in time_functions:
            for n_workers in workers:
                save_path = create_savefile_name(
                    data_dir,
                    time_func,
                    n_workers,
                    problem_name,
                    for_presentation,
                )

                if os.path.exists(save_path):
                    pbar.write(f"Rank data: Path exists: {save_path}")
                    pbar.update()
                    continue

                errors = D[time_func][n_workers][problem_name]

                ranks = compute_ranks(
                    errors=errors, pool=pool, n_bootstrap=n_bootstrap
                )

                np.savez(
                    save_path,
                    ranks=ranks,
                    problems=instance_problems,
                    n_bootstrap=n_bootstrap,
                    workers=workers,
                    time_functions=time_functions,
                )
                pbar.update()
