import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import numpy as np
import pandas as pd
import copy
import itertools


def plot_main(ga2m, X):
    """
    Visualize the effect of features on the target variable.

    Parameters
    ----------
    ga2m : CGA2M_plus.cga2m.Constraint_GA2M
        Trained model
    X : numpy.ndarray

    """
    num_use = len(ga2m.use_main_features)
    if num_use < 3:
        num_col = num_use
    else:
        num_col = 3
    num_row = int((num_use + 2) / 3)

    width = num_col * 4
    height = num_row * 3
    fig, axs = plt.subplots(nrows=num_row, ncols=num_col, figsize=(width, height))
    axs = axs.reshape(-1)

    for p, i in enumerate(ga2m.use_main_features):
        max_x = np.max(X[:, i])
        min_x = np.min(X[:, i])
        x = np.linspace(start=min_x, stop=max_x)
        a = ga2m.main_model_dict[i].predict(
            x.reshape(-1, 1), num_iteration=ga2m.main_model_dict[i].best_iteration
        )

        axs[p].set_title(i)
        axs[p].set_xlabel(r"$x_{}$".format(i), fontsize=12)
        axs[p].set_ylabel(r"$f_{}(x_{})$".format(i, i), fontsize=12)
        sns.lineplot(x=x, y=a, ax=axs[p])

    plt.tight_layout()
    plt.show()


def plot_interaction(ga2m, X, mode="3d"):
    """
    Visualize the effect of the pairs of features on the target variable.
    You can choose 2D or 3D.

    Parameters
    ----------
    ga2m : CGA2M_plus.cga2m.Constraint_GA2M
        Trained model

    X : numpy.ndarray

    mode : str
        if you set mode = '3d', this function displays a 3-dimensional graph.
        Otherwise, you will see a 2D graph.

    """
    if mode == "3d":
        plot_interaction_3d(ga2m, X)
    else:
        plot_interaction_2d(ga2m, X)


def plot_interaction_2d(ga2m, X):
    """
    Visualize the effect of the pairs of features on the target variable using a 2D graph.

    Parameters
    ----------
    ga2m : CGA2M_plus.cga2m.Constraint_GA2M
        Trained model

    X : numpy.ndarray
    """
    num_use = len(ga2m.use_interaction_features)
    if num_use < 3:
        num_col = num_use
    else:
        num_col = 3
    num_row = int((num_use + 2) / 3)

    width = num_col * 4
    hight = num_row * 3
    fig, axs = plt.subplots(nrows=num_row, ncols=num_col, figsize=(width, hight))
    axs = axs.reshape(-1)

    for p, (i, j) in enumerate(ga2m.use_interaction_features):
        max_x0 = np.max(X[:, i])
        min_x0 = np.min(X[:, i])
        max_x1 = np.max(X[:, j])
        min_x1 = np.min(X[:, j])
        x0 = np.linspace(start=min_x0, stop=max_x0)
        x1 = np.linspace(start=min_x1, stop=max_x1)

        a, b = np.meshgrid(x0, x1)
        x = np.hstack((a.reshape(-1, 1), b.reshape(-1, 1)))
        preds = ga2m.interaction_model_dict[(i, j)].predict(
            x.reshape(-1, 2),
            num_iteration=ga2m.interaction_model_dict[(i, j)].best_iteration,
        )

        cp = axs[p].contourf(a, b, preds.reshape(a.shape))
        plt.colorbar(cp, ax=axs[p])

        axs[p].set_title(r"$f_{0}._{1}(x_{0},x_{1})$".format(i, j))
        axs[p].set_xlabel(r"$x_{}$".format(i))
        axs[p].set_ylabel(r"$x_{}$".format(j))
    plt.tight_layout()
    plt.show()


def plot_interaction_3d(ga2m, X):
    """
    Visualize the effect of the pairs of features on the target variable using a 3D graph.

    Parameters
    ----------
    ga2m : CGA2M_plus.cga2m.Constraint_GA2M
        Trained model

    X : numpy.ndarray

    """
    num_use = len(ga2m.use_interaction_features)
    if num_use < 3:
        num_col = num_use
    else:
        num_col = 3
    num_row = int((num_use + 2) / 3)
    width = num_col * 4
    hight = num_row * 3
    fig = plt.figure(figsize=(width, hight))

    for p, (i, j) in enumerate(ga2m.use_interaction_features):
        axs = fig.add_subplot(num_row, num_col, p + 1, projection="3d")
        max_x0 = np.max(X[:, i])
        min_x0 = np.min(X[:, i])
        max_x1 = np.max(X[:, j])
        min_x1 = np.min(X[:, j])
        x0 = np.linspace(start=min_x0, stop=max_x0)
        x1 = np.linspace(start=min_x1, stop=max_x1)

        a, b = np.meshgrid(x0, x1)
        x = np.hstack((a.reshape(-1, 1), b.reshape(-1, 1)))
        preds = ga2m.interaction_model_dict[(i, j)].predict(
            x.reshape(-1, 2),
            num_iteration=ga2m.interaction_model_dict[(i, j)].best_iteration,
        )

        cp = axs.plot_surface(
            a, b, preds.reshape(a.shape), rstride=1, cstride=1, cmap=cm.coolwarm
        )

        axs.set_title(r"$f_{0}._{1}(x_{0},x_{1})$".format(i, j))
        axs.set_xlabel(r"$x_{}$".format(i))
        axs.set_ylabel(r"$x_{}$".format(j))
    plt.tight_layout()
    plt.show()


def show_importance(ga2m, after_prune=True, higher_mode=False):
    """
    Visualize the importance of features and the pair of features using bar plot.

    Parameters
    ----------
    ga2m : CGA2M_plus.cga2m.Constraint_GA2M
        Trained model

    after_prune : bool
        If True, this function displays the feature importance after pruning.

    higher_mode : bool
        If True, this function displays the feature importance which contain the effect of the higher-order term.

    """
    if after_prune:
        tmp_dict = copy.deepcopy(ga2m.after_feature_importance_)
    else:
        tmp_dict = copy.deepcopy(ga2m.before_feature_importance_)

    if higher_mode == False:
        del tmp_dict["higher"]

    impact_df = pd.DataFrame(
        tmp_dict.values(),
        index=[str(a) for a in tmp_dict.keys()],
        columns=["IMPORTANCE"],
    )
    sns.barplot(data=impact_df.T, orient="h")
    plt.title("IMPORTANCE")
    plt.show()
