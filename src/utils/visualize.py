import matplotlib.pyplot as plt

def plot_hist(values, title, xlabel, outfile):
    plt.figure()
    plt.hist(values, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_merge_curve(xs, ys, title, outfile):
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel("merge step")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
