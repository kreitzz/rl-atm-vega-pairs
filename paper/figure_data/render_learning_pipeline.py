"""Publication figures at their printed width, using saved research only."""
from pathlib import Path
import csv

BLUE = '#244b70'
TEAL = '#087f91'
GRAY = '#7d8791'
RED = '#ad554c'
PALE = '#edf3f7'
WIDTH = 7.1


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def one(data, **keys):
    found = [r for r in data if all(str(r[k]) == str(v) for k, v in keys.items())]
    assert len(found) == 1, keys
    return found[0]


def build(name, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9.5,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#8996a1'})

    def panel(height):
        fig, ax = plt.subplots(figsize=(WIDTH, height))
        return fig, ax

    def box(ax, xy, width, height, text, accent=False, size=10):
        from matplotlib.patches import FancyBboxPatch
        x, y = xy
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle='round,pad=0.008,rounding_size=0.015',
                     facecolor='#dff0f1' if accent else PALE,
                     edgecolor='#b9cbd7', linewidth=.8))
        ax.text(x+width/2, y+height/2, text, va='center', ha='center',
                fontsize=size, color=BLUE, linespacing=1.35)

    def arrow(ax, start, end, **kw):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.2, **kw))

    if name == 'learning_pipeline':
        fig, ax = panel(1.75)
        fig.subplots_adjust(left=.005, right=.995, top=.98, bottom=.02)
        ax.set(xlim=(0,1), ylim=(0,1)); ax.axis('off')
        for y, signal, result in [(.57, 'Relative put IV', 'Put-pair portfolio'),
                                  (.08, 'RV − IV persistence', 'Straddle portfolio')]:
            box(ax, (.01,y), .23,.34, signal)
            box(ax, (.29,y), .26,.34, 'Entry sizing\n50 / 75 / 100%')
            box(ax, (.60,y), .22,.34, result)
            arrow(ax,(.245,y+.17),(.28,y+.17));arrow(ax,(.555,y+.17),(.59,y+.17))
        box(ax, (.87,.265), .12,.47, 'Q policy\nCapital\nsplit', accent=True)
        arrow(ax,(.825,.74),(.865,.60));arrow(ax,(.825,.25),(.865,.39))

    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination / "learning_pipeline.pdf", facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    build("learning_pipeline", Path(__file__).resolve().parents[1] / "figures")
