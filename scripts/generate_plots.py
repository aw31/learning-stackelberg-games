import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

for SETTING in [1, 2]:
    print(f"Setting {SETTING}")

    clinch_data = pd.read_json(
        f"results_clinch_{SETTING}.jsonl", lines=True
    ).values.tolist()
    security_search_data = pd.read_json(
        f"results_security_search_{SETTING}.jsonl", lines=True
    ).values.tolist()

    # For security search data, average same keys
    security_search_data = [
        (key, np.mean([value for _, value in values]))
        for key, values in itertools.groupby(security_search_data, lambda x: x[0])
    ]

    plt.plot(*zip(*clinch_data))
    plt.xlabel("Number of targets")
    plt.ylabel("Calls to oracle")
    plt.title("Performance of Clinch")
    plt.show()

    x, y = zip(*clinch_data[5:])
    x = np.log(x)
    y = np.log(y)
    slope = np.polyfit(x, y, 1)[0]
    print(f"Clinch slope: {slope}")

    plt.plot(*zip(*security_search_data))
    plt.xlabel("Number of targets")
    plt.ylabel("Calls to oracle")
    plt.title("Performance of Security Search")
    plt.show()

    x, y = zip(*security_search_data[3:])
    x = np.log(x)
    y = np.log(y)
    slope = np.polyfit(x, y, 1)[0]
    print(f"Security Search slope: {slope}")

    sns.set_style("darkgrid")
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["figure.figsize"] = (5, 4)

    x, y = zip(*clinch_data)
    x2, y2 = zip(*security_search_data)

    coeffs = np.polyfit(np.log10(x), np.log10(y), 1)
    coeffs2 = np.polyfit(np.log10(x2), np.log10(y2), 1)

    fit_line = 10 ** (coeffs[1]) * x ** (coeffs[0])
    fit_line2 = 10 ** (coeffs2[1]) * x2 ** (coeffs2[0])

    plt.plot(x, y, label="ClinchSimplex")
    plt.plot(x2, y2, label="SecuritySearch")
    plt.plot(x, fit_line, "--", label=f"$y = c_1 N^{{{coeffs[0]:.2f}}}$")
    plt.plot(x2, fit_line2, "--", label=f"$y = c_2 N^{{{coeffs2[0]:.2f}}}$")

    plt.xlabel("Number of targets")
    plt.ylabel("Calls to oracle")
    plt.title(f"Clinch vs. Security Search (Setting {SETTING})")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().labelOnlyBase = False

    plt.savefig(f"plot_combined_setting_{SETTING}.png")  # Save plot
    plt.show()
