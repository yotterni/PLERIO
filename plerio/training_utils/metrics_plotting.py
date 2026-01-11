from pathlib import Path

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def pics_from_metrics(metrics_path: Path | str,
                      train_metrics: dict[str, list[float]],
                      val_metrics: dict[str, list[float]],
                      cell_line: str) -> None:
    train_df = pd.DataFrame().from_dict(train_metrics)
    val_df = pd.DataFrame().from_dict(val_metrics)

    train_df['fold'] = ['train'] * len(train_df)
    val_df['fold'] = ['val'] * len(val_df)

    df = pd.concat([train_df, val_df], axis=0)
    for metric_name in train_metrics:
        violin = sns.violinplot(data=df, x='fold', y=metric_name, hue='fold')
        # plt.legend()
        plt.title(cell_line)
        fig = violin.get_figure()
        fig.savefig(
            os.path.join(metrics_path, f'{metric_name}.png'))
        plt.clf()
