import matplotlib.pyplot as plt

def plot_2d_results(results, xlabel='fraction of zero-filled weights',
                    ylabel='error rate', xlog=False, ylog=False, title=None):
    """Create 2d plot of results given by sparsifying algorithm.

    Creates 2d plot with results' values on axes.

    :results: List of pairs to be plotted or dictionary with pairs as values
              (for any key). Both elements from pair must be recognized by
              matplotlib.pylot as numbers.
    :xlabel: Label for x axis.
    :ylabel: Label for y axis.
    :xlog: If true, x axis is logarithmic.
    :ylog: If true, y axis is logarithmic.
    :title: Title of the graph.
    """
    if type(results) is dict:
        results = results.values()
    zf_fraction_l = [res[0] for res in results]
    error_rate_l = [res[1] for res in results]
    plt.plot(zf_fraction_l, error_rate_l, 'ro')
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.show()