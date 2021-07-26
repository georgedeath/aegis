import os
import tqdm
import torch
import tqdm.auto
import matplotlib
import numpy as np

from matplotlib import pyplot as plt
from scipy.stats import median_abs_deviation, wilcoxon
from statsmodels.stats.multitest import multipletests

from . import test_problems, util


def read_in_results(
    time_functions, workers, problems, method_names, n_runs, budget
):
    # D[time_func][workers][problem][method]
    D = {}

    total = (
        len(time_functions)
        * len(workers)
        * len(method_names)
        * len(problems)
        * n_runs
    )

    with tqdm.auto.tqdm(total=total, leave=True) as pbar:
        for time_func in time_functions:
            D[time_func] = {}

            for n_workers in workers:
                D[time_func][n_workers] = {}

                for problem_name, problem_params in problems:
                    pn = problem_name
                    if "d" in problem_params:
                        pn = f"{pn:s}{problem_params['d']:d}"

                    if problem_name not in D[time_func][n_workers]:
                        D[time_func][n_workers][pn] = {}

                    f_class = getattr(test_problems, problem_name)
                    f = f_class(**problem_params)

                    for method_name in method_names:
                        res = np.zeros((n_runs, budget))
                        acq_params = {}

                        if "-" in method_name:
                            mn, eps_or_acq, *eta = method_name.split("-")

                            # only for aegis methods
                            if "aegis" in method_name:
                                if len(eta) == 0:
                                    acq_params["eta"] = 0.5
                                else:
                                    acq_params["eta"] = float(eta[0])

                                if isinstance(eps_or_acq, str):
                                    acq_params["epsilon"] = eps_or_acq
                                else:
                                    acq_params["epsilon"] = float(eps_or_acq)

                            elif "BatchBO" in method_name:
                                acq_params["acq_name"] = eps_or_acq

                            else:
                                err = f"Invalid method name: {method_name:s}"
                                raise ValueError(err)

                        else:
                            mn = method_name

                        for i, run_no in enumerate(range(1, n_runs + 1)):
                            fn = util.generate_save_filename(
                                time_func,
                                problem_name,
                                budget,
                                n_workers,
                                mn,
                                run_no,
                                problem_params,
                                acq_params,
                            )

                            try:

                                data = torch.load(fn)
                                Ytr = data["Ytr"].numpy().ravel()
                                n = Ytr.size

                                res[i, :n] = Ytr

                                if n != budget:
                                    print("Not full:", fn, Ytr.shape)

                            except FileNotFoundError:
                                print("Missing", os.path.basename(fn))
                                raise
                            except:  # noqa: E722
                                print(method_name)
                                print(mn)
                                print(fn)
                                raise

                            pbar.update()

                        res = np.abs(res - f.yopt.ravel()[0])
                        res = np.minimum.accumulate(res, axis=1)  # type: ignore

                        D[time_func][n_workers][pn][method_name] = res

    return D


def results_plot_maker(
    ax,
    yvals,
    xvals,
    xlabel,
    ylabel,
    title,
    colors,
    LABEL_FONTSIZE,
    TITLE_FONTSIZE,
    TICK_FONTSIZE,
    use_fill_between=True,
    fix_ticklabels=False,
):
    # here we assume we're plotting to a matplotlib axis object
    # and yvals is a LIST of arrays of size (n_runs, iterations),
    # where each can be different sized
    # and if xvals is given then len(xvals) == len(yvals)

    # set the labelling
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    for color, x, Y in zip(colors, xvals, yvals):
        bot, mid, top = [
            _.ravel() for _ in np.percentile(Y, [25, 50, 75], axis=0)
        ]

        if use_fill_between:
            ax.fill_between(x, bot.flat, top.flat, color=color, alpha=0.15)

        ax.plot(x, mid, color=color)
        ax.plot(x, bot, "--", color=color, alpha=0.15)
        ax.plot(x, top, "--", color=color, alpha=0.15)

    # set the xlim
    min_x = np.min([np.min(x) for x in xvals])
    max_x = np.max([np.max(x) for x in xvals])
    ax.set_xlim([0, max_x])

    ax.axvline(min_x, linestyle="dashed", color="gray", linewidth=1, alpha=0.5)

    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FONTSIZE)

    # set the alignment for outer ticklabels
    ax.set_xticks([0, 50, 100, 150, 200])
    if fix_ticklabels:
        ticklabels = ax.get_xticklabels()
        if len(ticklabels) > 0:
            ticklabels[0].set_ha("left")
            ticklabels[-1].set_ha("right")

    ax.yaxis.set_major_formatter(
        matplotlib.ticker.StrMethodFormatter("{x:>4.1f}")
    )


def make_conv_plots(
    D,
    time_func,
    problems,
    workers,
    budget,
    method_names,
    method_cols,
    fname_prefix=None,
    TITLE_FONTSIZE=19,
    LABEL_FONTSIZE=18,
    TICK_FONTSIZE=16,
    savefig=False,
):
    for problem_name, problem_params in problems:
        pn = problem_name
        if "d" in problem_params:
            pn = f"{pn:s}{problem_params['d']:d}"

        f_class = getattr(test_problems, problem_name)
        f = f_class(**problem_params)

        start = 2 * f.dim - 1
        end = budget

        x = np.arange(start + 1, end + 1)

        n = len(workers)

        fig, ax = plt.subplots(1, n, figsize=(8, 2), sharey=True, dpi=120)
        for i in range(n):
            n_workers = workers[i]
            a = ax[i]

            yvals = []
            xvals = []

            for method_name in method_names:
                Y = D[time_func][n_workers][pn][method_name][:, start:end]
                Y = np.log(Y)

                yvals.append(Y)
                xvals.append(x)

            title = f"{pn:s}, q={n_workers:d}"

            ylabel = r"$\log(R_t)$" if i == 0 else None
            xlabel = "$t$"

            results_plot_maker(
                a,
                yvals,
                xvals,
                xlabel,
                ylabel,
                title,
                method_cols,
                LABEL_FONTSIZE,
                TITLE_FONTSIZE,
                TICK_FONTSIZE,
                use_fill_between=True,
                fix_ticklabels=True,
            )

            a.get_xaxis().set_label_coords(0.5, -0.15)
            # ensure labels are all in the same place!
            a.get_yaxis().set_label_coords(-0.22, 0.5)

            if i > 0:
                a.yaxis.set_ticks_position("none")

        plt.subplots_adjust(
            left=0, right=1, bottom=0, top=1, wspace=0.03, hspace=0
        )

        if savefig:
            fname = f"{pn}.pdf"
            if fname_prefix is not None:
                fname = f"{fname_prefix}_{fname}"

            plt.savefig(fname, bbox_inches="tight")

        plt.show()


def make_legend(
    method_paper_names,
    method_cols,
    onecol_inds,
    twocol_inds,
    fname_prefix=None,
    savefig=False,
):
    nmethods = len(method_paper_names)

    for namecols, ncols in [
        ("onecol", int(np.ceil(nmethods / 2))),
        ("twocol", nmethods),
    ]:
        if namecols == "onecol":
            h = 2
        else:
            h = 1

        fig, ax = plt.subplots(1, 1, figsize=(19, h))
        for method_name, color in zip(method_paper_names, method_cols):
            ax.plot([0, 1], [0, 1], color=color, label=method_name)

        legend_options = {
            "loc": 3,
            "framealpha": 1,
            "frameon": False,
            "fontsize": 20,
            "handletextpad": 0.3,
            "columnspacing": 1,
            "ncol": ncols,
        }

        legend = ax.legend(**legend_options)

        if namecols == "onecol":
            inds = onecol_inds

        else:
            inds = twocol_inds

        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[i] for i in inds]
        labels = [labels[i] for i in inds]
        legend = ax.legend(handles, labels, **legend_options)

        # increase legend line widths
        for legobj in legend.legendHandles:
            legobj.set_linewidth(5.0)

        # remove all plotted lines
        for _ in range(len(ax.lines)):
            ax.lines.pop(0)

        fig.canvas.draw()
        bbox = legend.get_window_extent()
        bbox = bbox.from_extents(*[bbox.extents + np.array([-5, -5, 5, 5])])
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

        ax.axis("off")

        fname = f"legend_{namecols}.pdf"
        if fname_prefix is not None:
            fname = f"{fname_prefix}_{fname}"

        if savefig:
            fig.savefig(fname, dpi="figure", bbox_inches=bbox)

        plt.show()


def create_table_data(
    results, workers, problems, method_names, n_exps, time=-1
):
    """

    """
    method_names = np.array(method_names)
    n_means = len(method_names)

    # table_data[problem_name] = {'median', 'MAD', 'stats_equal_to_best_mask'}
    table_data = {}

    for n_workers in workers:
        table_data[n_workers] = {}

        for problem_name, problem_params in problems:
            pn = problem_name
            if "d" in problem_params:
                pn = f"{pn:s}{problem_params['d']:d}"

            best_seen_values = np.zeros((n_means, n_exps))

            for i, mean_name in enumerate(method_names):
                # best seen evaluate at the end of the optimisation run
                best_seen_values[i, :] = results[n_workers][pn][mean_name][
                    :, time
                ]

            medians = np.median(best_seen_values, axis=1)
            MADS = median_abs_deviation(
                best_seen_values, scale="normal", axis=1
            )

            # best method -> lowest median value
            best_method_idx = np.argmin(medians)

            # mask of methods equivlent to the best
            stats_equal_to_best_mask = np.zeros(n_means, dtype="bool")
            stats_equal_to_best_mask[best_method_idx] = True

            # perform wilcoxon signed rank test between best and all other methods
            p_values = []
            for i, mean_name in enumerate(method_names):
                if i == best_method_idx:
                    continue
                # a ValueError will be thrown if the runs are all identical,
                # therefore we can assign a p-value of 0 as they are identical
                try:
                    _, p_value = wilcoxon(
                        best_seen_values[best_method_idx, :],
                        best_seen_values[i, :],
                    )
                    p_values.append(p_value)

                except ValueError:
                    p_values.append(0)

            # calculate the Holm-Bonferroni correction
            reject_hyp, pvals_corrected, _, _ = multipletests(
                p_values, alpha=0.05, method="holm"
            )

            for reject, mean_name in zip(
                reject_hyp,
                [
                    m
                    for m in method_names
                    if m != method_names[best_method_idx]
                ],
            ):
                # if we can't reject the hypothesis that a technique is
                # statistically equivalent to the best method
                if not reject:
                    idx = np.where(np.array(method_names) == mean_name)[0][0]
                    stats_equal_to_best_mask[idx] = True

            # store the data
            table_data[n_workers][pn] = {
                "medians": medians,
                "MADS": MADS,
                "stats_equal_to_best_mask": stats_equal_to_best_mask,
            }

    return table_data


def create_table(
    table_data,
    n_workers,
    problem_rows,
    problem_paper_rows,
    problem_dim_rows,
    method_names,
    method_names_for_table,
    caption="",
):
    """

    """

    head = r"""
\begin{table*}[t]
\setlength{\tabcolsep}{2pt}
\newcolumntype{z}{>{\small}S}
\sisetup{table-format=1.2e-1,table-number-alignment=center}"""
    head += "\n" + r"\caption{%s}" % caption
    head += (
        "\n"
        + r"""\resizebox{1\textwidth}{!}{%
\begin{tabular}{l Sz Sz Sz Sz Sz}"""
    )

    foot = r""" \end{tabular}
}
\label{tbl:synthetic_results}
\end{table*}"""

    print(head)
    for probs, probs_paper, probs_dim in zip(
        problem_rows, problem_paper_rows, problem_dim_rows
    ):

        print(r"\toprule")
        print(r"    \bfseries Method")

        # column titles: Problem name (dim).
        print_string = ""
        for prob, dim in zip(probs_paper, probs_dim):
            print_string += r"    & \multicolumn{2}{c"
            print_string += r"}{\bfseries "
            print_string += r"{:s} ({:d})".format(prob, dim)
            print_string += "} \n"

        print_string = print_string[:-2] + " \\\\ \n"

        # column titles: Median MAD
        for prob in probs:
            print_string += r"    & \multicolumn{1}{c}{Median}"
            print_string += r" & \multicolumn{1}{c"
            print_string += "}{MAD}\n"
        print_string = print_string[:-1] + "  \\\\ \\midrule"
        print(print_string)

        # results printing
        for i, (method_name, method_name_table) in enumerate(
            zip(method_names, method_names_for_table)
        ):
            print_string = "    "
            print_string += method_name_table + " & "

            # table_data[problem_name] = {'median', 'MAD', 'stats_equal_to_best_mask'}
            for prob in probs:
                med = "{:4.2e}".format(
                    table_data[n_workers][prob]["medians"][i]
                )
                mad = "{:4.2e}".format(table_data[n_workers][prob]["MADS"][i])

                best_methods = table_data[n_workers][prob][
                    "stats_equal_to_best_mask"
                ]
                best_idx = np.argmin(table_data[n_workers][prob]["medians"])

                if i == best_idx:
                    med = r"\best " + med
                    mad = r"\best " + mad

                elif best_methods[i]:
                    med = r"\statsimilar " + med
                    mad = r"\statsimilar " + mad

                print_string += med + " & " + mad + " & "

            print_string = print_string[:-2] + "\\\\"
            print(print_string)

        print("\\bottomrule")

    print(foot)
    print()


def make_equaltobest_plot(
    table_data,
    method_names,
    problems,
    workers,
    TICK_FONTSIZE=18,
    LEGEND_FONTSIZE=20,
    TITLE_FONTSIZE=20,
    legendloc=None,
    plotheight=None,
    fname_prefix=None,
    savefig=False,
):
    nmethods = len(method_names)
    width = 0.6 / len(workers)

    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(
        1, 1, figsize=(8, 4 if plotheight is None else plotheight), dpi=120
    )
    for i, (n_workers, color) in enumerate(zip(workers, colors)):
        best_counts = np.zeros(nmethods)

        c = 0

        for pn in table_data[n_workers]:
            best_counts += table_data[n_workers][pn][
                "stats_equal_to_best_mask"
            ]
            c += 1

        assert c == len(problems), (c, len(problems))

        best_counts /= c

        x = np.arange(nmethods) + i * width

        tick_labels = None
        if (i == 1) or len(workers) == 1:
            tick_labels = method_names

        _ = ax.barh(
            x,
            width=best_counts,
            height=width,
            tick_label=tick_labels,
            zorder=2,
            color=color,
        )

        # fake plot to create a better legend
        ax.plot(0, 0, label=f"q = {n_workers:d}", color=color)

    legend_options = {
        "loc": "lower right" if legendloc is None else legendloc,
        "fontsize": LEGEND_FONTSIZE,
        "ncol": 1,
        "labelspacing": 0.1,
        "handletextpad": 0.4,
    }
    leg = ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, **legend_options)
    for line in leg.get_lines():
        line.set_linewidth(5)

    ax.set_xlabel("Proportion of functions", fontsize=TICK_FONTSIZE)
    ax.set_title(
        "Proportion of times each method is the best performing",
        fontsize=TITLE_FONTSIZE,
    )
    ax.set_xlim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="x", zorder=1, alpha=0.5)

    ax.tick_params(axis="x", which="major", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", which="major", labelsize=TICK_FONTSIZE)

    ticklabels = ax.get_xticklabels()
    if len(ticklabels) > 0:
        ticklabels[0].set_ha("left")
        ticklabels[-1].set_ha("right")

    if savefig:
        fname = "equaltobest.pdf"
        if fname_prefix is not None:
            fname = f"{fname_prefix}_{fname}"
        plt.savefig(fname, bbox_inches="tight")
    plt.show()


def plot_solo_q(
    D,
    time_func,
    problems,
    n_workers,
    budget,
    method_names,
    method_cols,
    fname=None,
    TITLE_FONTSIZE=19,
    LABEL_FONTSIZE=18,
    TICK_FONTSIZE=16,
    savefig=False,
):
    fig, ax = plt.subplots(1, 5, figsize=(16, 2), sharey=False, dpi=120)

    for i, (problem_name, problem_params) in enumerate(problems):
        pn = problem_name
        if "d" in problem_params:
            pn = f"{pn:s}{problem_params['d']:d}"

        f_class = getattr(test_problems, problem_name)
        f = f_class(**problem_params)

        start = 2 * f.dim - 1
        end = budget

        x = np.arange(start + 1, end + 1)

        a = ax[i]

        yvals = []
        xvals = []

        for method_name in method_names:
            Y = D[time_func][n_workers][pn][method_name][:, start:end]
            Y = np.log(Y)

            yvals.append(Y)
            xvals.append(x)

            title = f"{pn:s}, q={n_workers:d}"

            ylabel = r"$\log(R_t)$" if i == 0 else None
            xlabel = "$t$"

            results_plot_maker(
                a,
                yvals,
                xvals,
                xlabel,
                ylabel,
                title,
                method_cols,
                LABEL_FONTSIZE,
                TITLE_FONTSIZE,
                TICK_FONTSIZE,
                use_fill_between=True,
                fix_ticklabels=True,
            )

        a.get_xaxis().set_label_coords(0.5, -0.15)
        # ensure labels are all in the same place!
        a.get_yaxis().set_label_coords(-0.22, 0.5)

    plt.subplots_adjust(
        left=0, right=1, bottom=0, top=1, wspace=0.25, hspace=0
    )

    if savefig:
        fname = f"{fname}.pdf"
        plt.savefig(fname, bbox_inches="tight")

    plt.show()
