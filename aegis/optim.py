import os
import torch
import numpy as np
from . import gp, test_problems, transforms, util, executor, time_dists, batch


def perform_optimisation(
    problem_name,
    problem_params,
    run_no,
    budget,
    n_workers,
    acq_name,
    acq_params,
    time_name,
    save_every=10,
    repeat_no=None,
):

    # set up the saving paths
    save_path = util.generate_save_filename(
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
        load_path = save_path
        print("Loading saved run")
    else:
        load_path = util.generate_data_filename(
            problem_name, run_no, problem_params, repeat_no=repeat_no
        )

    # load the training data
    data = torch.load(load_path)
    Xtr = data["Xtr"]
    Ytr = data["Ytr"]

    # if it has additional arguments add them to the dictionary passed to f
    if "problem_params" in data:
        problem_params.update(data["problem_params"])

    print(f"Training data shape: {Xtr.shape}")

    # load the problem instance
    f = getattr(test_problems, problem_name)(**problem_params)

    # wrap the problem for torch and so that it resides in [0, 1]^d
    f = util.TorchProblem(util.UniformProblem(f))

    # instantiate the time function
    time_class = getattr(time_dists, time_name)
    time_func = time_class()

    # get the acquisition function class
    acq_class = getattr(batch, acq_name)

    # run the BO
    asbo = AsyncBO(
        f,
        Xtr,
        Ytr,
        acq_class,
        acq_params,
        budget,
        n_workers,
        q=1,
        time_func=time_func,
        verbose=True,
    )

    # useful stuff to keep a record of
    save_dict = {
        "problem_name": problem_name,
        "problem_params": problem_params,
        "q": 1,
        "n_workers": n_workers,
        "acq_name": acq_name,
        "time_name": time_name,
        "budget": budget,
    }

    while not asbo.finished:
        asbo.step()

        # save every so often or when we've finished
        if asbo.Xtr.shape[0] % save_every == 0 or asbo.finished:
            # get the results so far
            res = asbo.get_results()
            save_dict.update(res)

            torch.save(obj=save_dict, f=save_path)
            print(f"Saving: {save_path:s}")

    print("Finished run")


class AsyncBO:
    def __init__(
        self,
        func,
        Xtr,
        Ytr,
        acq_class,
        acq_params,
        budget,
        n_workers,
        q=1,
        time_func=time_dists.halfnorm(),
        verbose=False,
    ):
        self.f = func
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.acq_class = acq_class
        self.acq_params = acq_params
        self.budget = budget
        self.n_workers = n_workers
        self.q = q
        self.output_transform = getattr(transforms, "Transform_Standardize",)
        self.time_func = time_func
        self.verbose = verbose

        self.interface = executor.SimExecutorJumpToCompletedJob(
            n_workers=self.n_workers, time_func=time_func, verbose=False,
        )

        self.dtype = Ytr.dtype

        self.ue = util.UnderEval(self.n_workers, self.f.dim, self.dtype)

        # time storage
        self.time = torch.zeros_like(Ytr, dtype=self.dtype)
        self.time_taken = torch.zeros_like(Ytr, dtype=self.dtype)

        # counters
        self.n_submitted = Xtr.shape[0]
        self.n_completed = Xtr.shape[0]

        # current iteration number
        self.init_size = Xtr.shape[0]

        # bounds
        self.ls_bounds = [1e-4, np.sqrt(Xtr.shape[1])]
        self.out_bounds = [1e-4, 10]

        # acquisition function, initialised after we have built a model
        self._acq = None

    @property
    def acq(self):
        if self._acq is None:
            self._acq = self.acq_class(
                model=self.model,
                lb=self.f.lb,
                ub=self.f.ub,
                under_evaluation=self.ue.get(),
                **self.acq_params,
            )

        return self._acq

    @property
    def finished(self):
        return self.n_completed == self.budget

    def _process_completed_jobs(self):
        # get the completed jobs
        completed_jobs = self.interface.get_completed_jobs()
        n = len(completed_jobs)

        # storage, so we only cat once
        _Xtr = torch.zeros((n, self.f.dim), dtype=self.dtype)
        _Ytr = torch.zeros(n, dtype=self.dtype)
        _time_taken = torch.zeros(n, dtype=self.dtype)
        _time = torch.zeros(n, dtype=self.dtype)

        # add the results to the training data
        for i, job in enumerate(completed_jobs):
            _Xtr[i] = job["x"]
            _Ytr[i] = job["y"]
            _time_taken[i] = torch.as_tensor(job["t"], dtype=self.dtype)
            _time[i] = self.interface.status["t"]

            # remove from under evaluation
            self.ue.remove(_Xtr[i])

            if self.verbose:
                print(
                    f"Completed -> f(x): {_Ytr[i]:0.3f}",
                    f"time taken: {_time_taken[i]:0.3f}",
                )

        # store the results
        self.Xtr = torch.cat((self.Xtr, _Xtr))
        self.Ytr = torch.cat((self.Ytr, _Ytr))
        self.time_taken = torch.cat((self.time_taken, _time_taken))
        self.time = torch.cat((self.time, _time))

        # add to the completed jobs counter
        self.n_completed += len(completed_jobs)

    def _create_and_submit_jobs(self):
        # only submit up to the budget
        n_to_submit = np.minimum(
            self.budget - self.n_submitted,  # total left to submit
            self.interface.status["n_free_workers"],  # free workers
        )
        if n_to_submit < 1:
            return

        # update the acquisition function ready for use.
        self.acq.update(self.model, self.ue.get())

        # get locations to evaluate from the acquisition function, create a
        # job, and submit it
        for _ in range(n_to_submit):
            # get the next location to evaluate
            x = self.acq.get_next()

            # create the job and submit it
            job = {"x": x, "f": self.f}
            self.interface.add_job_to_queue(job)

            # add to under evaluation
            self.ue.add(x)

            if self.verbose:
                print("Submitted ->", x.numpy().ravel())

        self.n_submitted += n_to_submit

    def step(self):
        if self.verbose:
            print(
                f" Submitted: {self.n_submitted} / {self.budget}"
                f" Completed: {self.n_completed} / {self.budget}"
            )

        # if the optimisation is complete, do nothing.
        if self.finished:
            print("finished")
            return

        # run the jobs until we there's one free
        if self.n_submitted < self.budget:
            self.interface.run_until_n_free(self.q)

        # if we've submitted our budget's worth of jobs, keep waiting
        # until one more worker each time is free. i.e. 1, 2, ..., n_workers
        else:
            self.interface.run_until_n_free(
                self.n_workers - (self.budget - self.n_completed) + 1
            )

        if self.verbose:
            print("Time:", self.interface.status["t"])

        # get the completed jobs
        self._process_completed_jobs()

        # if we've submitted our budget's worth of jobs,
        # do not submit any more
        if self.n_submitted == self.budget:
            return

        # scale the output
        T_out = self.output_transform(self.Ytr)
        train_y = T_out.scale_mean(self.Ytr)
        train_x = self.Xtr

        # fit the gp
        self.model, self.likelihood = gp.create_and_fit_GP(
            train_x=train_x,
            train_y=train_y,
            ls_bounds=self.ls_bounds,
            out_bounds=self.out_bounds,
            n_restarts=10,
            verbose=self.verbose,
        )
        self.model.eval()

        # submit jobs
        self._create_and_submit_jobs()

        if self.verbose:
            print("------------------------------")
            print()

    def get_results(self):
        resd = {
            "Xtr": self.Xtr,
            "Ytr": self.Ytr,
            "time_taken": self.time_taken,
            "time": self.time,
        }

        return resd
