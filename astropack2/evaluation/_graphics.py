import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np

from math import log10, floor

label_fontdict = {"family": "sans-serif", "weight": "normal", "size": 10}

legend_label_fontdict = {"family": "sans-serif", "weight": "normal", "size": 10}

legend_title_fontdict = {"family": "sans-serif", "weight": "bold", "size": 10}

tick_size = 20

def plot_test_graphs(
    predictions, true_values, bins, cmap_name, param_string, param_unit, n_ticks, legend
):
    if not bins:
        bins = [min(true_values) - 1, max(true_values) + 1]

    df = pd.merge(left=true_values, left_index=True, right=predictions, right_index=True)

    df.columns = ["TRUE_VALUE", "PREDICTION"]
    df["ERROR"] = df["PREDICTION"] - df["TRUE_VALUE"]

    fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[0.6, 0.4], figsize=(15, 6))

    colors = mpl.colormaps[cmap_name](np.linspace(0.5, 1.00, len(bins) - 1))

    bins_intervals = []
    for bin_index in range(0, len(bins) - 1):
        bin_min = bins[bin_index]
        bin_max = bins[bin_index + 1]
        bins_intervals.append(f"[{bin_min} {param_unit}, {bin_max} {param_unit}]")

        df_bin = df[(df["TRUE_VALUE"] >= bin_min) & (df["TRUE_VALUE"] < bin_max)].copy()

        sns.scatterplot(
            data=df_bin,
            x="PREDICTION",
            y="TRUE_VALUE",
            ax=ax[0],
            s=9,
            color=colors[bin_index],
            linewidth=0,
            zorder=2,
        )
        kde = sns.kdeplot(data=df_bin, x="ERROR", ax=ax[1], color=colors[bin_index])

    handles = [plot_handles(ax[0], "s", colors[i]) for i in range(len(bins_intervals))]

    min_lim_x = round_to_n(bins[0] - (bins[-1] - bins[0]) * 0.05, 1)
    max_lim_x = round_to_n(bins[-1] + (bins[-1] - bins[0]) * 0.05, 1)

    ax[0].plot(
        [min_lim_x, max_lim_x],
        [min_lim_x, max_lim_x],
        ls="--",
        lw=1.5,
        color="k",
        zorder=3,
    )

    ax[0] = beautify_graph(
        ax=ax[0],
        x_limits=[min_lim_x, max_lim_x],
        y_limits=[min_lim_x, max_lim_x],
        x_n_ticks=n_ticks,
        y_n_ticks=n_ticks,
        x_label=f"Predicted {param_string}",
        y_label=f"True {param_string}",
        grid = True
    )

    min_lim_x = round_to_n(-(df["ERROR"].abs().median() * 20), 1)
    max_lim_x = round_to_n((df["ERROR"].abs().median() * 20), 1)

    min_lim_y = 0

    y_maxes = []
    for line in kde.lines:
        x, y = line.get_data()
        y_maxes.append(max(y))

    max_lim_y = max(y_maxes) * 1.1

    ax[1].plot([0, 0], [0, max_lim_y], ls="--", lw=1.5, color="k", zorder=3)

    ax[1] = beautify_graph(
        ax=ax[1],
        x_limits=[min_lim_x, max_lim_x],
        y_limits=[min_lim_y, max_lim_y],
        x_n_ticks=n_ticks,
        y_n_ticks=n_ticks,
        x_label="Error",
        y_label="Density",
        grid = True
    )

    ax[1].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    if legend:
        leg = fig.legend(
            handles,
            bins_intervals,
            title=f"{param_string}",
            title_fontproperties=legend_title_fontdict,
            ncols=len(bins_intervals),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.075),
            framealpha=1,
            prop=legend_label_fontdict,
            markerscale=3,
            borderpad=1,
        )

        leg._legend_box.sep = 20

    return fig


def plot_comparison_graph(results, metric, error, cmap_name, param_unit, n_ticks, legend):
    fig = plt.figure(figsize = (3 * len(results[next(iter(results))]), 5))
    ax = fig.add_axes([0,0,1,1])

    n_bars = len(results)
    bar_width = (1 - 1/n_bars)/n_bars
    
    hatches = [''] * int(n_bars/2) + ['/'] * int(n_bars/2) 
    paddings = np.arange(0, n_bars * bar_width, bar_width) - (bar_width/2 *  (n_bars - 1))
    colors = mpl.colormaps[cmap_name](np.linspace(0.25, 0.75, n_bars))

    for index, key in enumerate(results):
        ax.bar(x = results[key].index + paddings[index],
               height = results[key][metric],
               yerr = results[key][error],
               width = bar_width,
               color = colors[index],
               hatch = hatches[index],
               edgecolor = 'k',
               linewidth = 2.5,
               capsize = 5,
               error_kw = {'elinewidth': 3},
               label = key,
               zorder = 2)

    min_lim_y = 0
    max_lim_y = max([(x[metric] + x[error]).max() for x in list(results.values())])

    ax = beautify_graph(ax = ax,
                        x_limits = None,
                        y_limits = [min_lim_y, round_to_n(max_lim_y + (max_lim_y - min_lim_y) * 0.1, 2)],
                        x_n_ticks = None,
                        y_n_ticks = n_ticks,
                        x_label = 'Parameter Interval',
                        y_label = f'MAD ({param_unit})')
    
    ax.set_xticks(ticks = results[key].index,
                  labels = results[key]['bin'])
    
    ax.grid(axis = 'y', zorder = 0)

    if legend:
        leg = ax.legend(title = 'Features',
                        title_fontproperties = legend_title_fontdict,
                        prop = legend_label_fontdict,
                        framealpha=1,
                        handlelength=3,
                        handleheight=1.5,
                        borderpad=1,
                        bbox_to_anchor = (1.01, 1))
        
        leg._legend_box.sep = 20

    return fig

def beautify_graph(ax, x_limits, y_limits, x_n_ticks, y_n_ticks, x_label, y_label, grid = None):
    if x_limits:
        ax.set_xlim(x_limits[0], x_limits[1])
        ax.set_xticks(np.linspace(x_limits[0], x_limits[1], x_n_ticks))

    if y_limits:
        ax.set_ylim(y_limits[0], y_limits[1])
        ax.set_yticks(np.linspace(y_limits[0], y_limits[1], y_n_ticks))

    ax.tick_params(labelsize=tick_size)

    ax.set_xlabel(x_label, fontdict=label_fontdict, labelpad=15)
    ax.set_ylabel(y_label, fontdict=label_fontdict, labelpad=15)

    if grid:
        ax.grid(zorder=0)

    return ax


def round_to_n(x, n):
    return round(x, -int(floor(log10(abs(x)))) + n - 1)


def plot_handles(ax, m, c):
    handle = ax.plot([], [], marker=m, color=c, ls="None")[0]
    return handle


def plot_regression_with_residuals(y_true, y_pred, bins=None, param_name=None, param_unit=None, cmap=None, point_size=3):
    """
    Gera um gráfico padrão para avaliação de regressão:
    - Painel principal: dispersão y_pred vs y_true, linha de identidade, R² e MAD.
    - Painel superior: resíduos vs y_true, linhas de ±3σ e porcentagem de objetos dentro desse intervalo.
    - Suporte a coloração por bins ou contínua.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    # Detectar parâmetro e ajustar padrões
    param_map = {
        'teff': {
            'name': 'Teff',
            'unit': 'K',
            'cmap': 'Reds',
            'xlabel': 'Teff True (K)',
            'ylabel': 'Teff Predicted (K)'
        },
        'feh': {
            'name': '[Fe/H]',
            'unit': 'dex',
            'cmap': 'Blues',
            'xlabel': '[Fe/H] True (dex)',
            'ylabel': '[Fe/H] Predicted (dex)'
        },
        'logg': {
            'name': 'logg',
            'unit': 'dex',
            'cmap': 'Greens',
            'xlabel': 'logg True (dex)',
            'ylabel': 'logg Predicted (dex)'
        }
    }

    # Detectar param automaticamente
    param_key = None
    if param_name is not None:
        pname = param_name.strip().lower().replace('[', '').replace(']', '').replace('/', '').replace(' ', '')
        if 'teff' in pname:
            param_key = 'teff'
        elif 'feh' in pname or 'feh' in pname.replace('[', '').replace(']', '').replace('/', '').replace(' ', ''):
            param_key = 'feh'
        elif 'logg' in pname:
            param_key = 'logg'

    if param_key in param_map:
        pinfo = param_map[param_key]
    else:
        # fallback para Teff
        pinfo = param_map['teff']

    # Sobrescrever argumentos se não definidos
    if param_unit == "K" or param_unit is None:
        param_unit = pinfo['unit']
    if cmap == "Reds" or cmap is None:
        cmap = pinfo['cmap']
    # Labels
    xlabel = pinfo['xlabel']
    ylabel = pinfo['ylabel']

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_pred - y_true
    sigma = np.std(residuals)
    mad = np.median(np.abs(residuals))
    # R² Score
    try:
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
    except ImportError:
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)

    # Painel de resíduos
    ax_res = fig.add_subplot(gs[0])
    if bins is not None:
        c = np.digitize(y_true, bins)
        sc_res = ax_res.scatter(y_true, residuals, c=c, cmap=cmap, s=point_size, alpha=0.7)
    else:
        sc_res = ax_res.scatter(y_true, residuals, c=y_true, cmap=cmap, s=point_size, alpha=0.7)
    ax_res.axhline(0, color='k', linestyle='--', linewidth=1)
    ax_res.axhline(3*sigma, color='k', linestyle='--', linewidth=1)
    ax_res.axhline(-3*sigma, color='k', linestyle='--', linewidth=1)
    ax_res.set_ylabel("Residuals")
    ax_res.set_xticks([])
    # Criar handle manualmente apenas para a linha tracejada (--- ±3σ)
    from matplotlib.lines import Line2D
    line_handle = Line2D([0], [0], color='k', linestyle='--', linewidth=1)
    ax_res.legend([line_handle], ["±3σ"], loc="lower left", fontsize=8)

    # Painel principal
    ax_main = fig.add_subplot(gs[1])
    if bins is not None:
        c = np.digitize(y_true, bins)
        sc = ax_main.scatter(y_true, y_pred, c=c, cmap=cmap, s=point_size, alpha=0.7)
        # Adiciona colorbar com os intervalos
        cmap_obj = cm.get_cmap(cmap)
        n_colors = getattr(cmap_obj, 'N', 256)
        norm = mcolors.BoundaryNorm(bins, n_colors)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=ax_main,
            orientation='vertical',
            pad=0.02,
            aspect=60
        )
        if bins is not None and len(bins) > 1:
            tick_locs = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
            cbar.set_ticks(tick_locs)
            cbar.ax.set_yticklabels([f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)])
        cbar.set_label(f"{pinfo['name']} intervals ({pinfo['unit']})", fontsize=10)
    else:
        sc = ax_main.scatter(y_true, y_pred, c=y_true, cmap=cmap, s=point_size, alpha=0.7)
    minv, maxv = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax_main.plot([minv, maxv], [minv, maxv], 'k-', lw=1)
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.text(0.05, 0.95, f"R² Score: {r2:.4f}", transform=ax_main.transAxes, fontsize=9, va='top')
    ax_main.text(0.95, 0.05, f"MAD: {mad:.2f} {pinfo['unit']}", transform=ax_main.transAxes, fontsize=9, ha='right', va='bottom')

    plt.tight_layout()
    return fig
