import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def create_chart(x: pd.DataFrame, name: str, save_path: str = None):

    col_names = x.columns
    plt.title(f'Benchmark of {name} kernel')
    plt.xlabel(col_names[0])
    plt.ylabel('GB/s')
    plt.plot(x[col_names[0]], x[[*col_names[-2:]]], label=[*col_names[-2:]])
    plt.legend()
    plt.show()

    if save_path:
        plt.savefig(save_path)

if __name__ == '__main__':
    """
    Folder structure expected:
        - kernel_name
            - performance.csv
    """

    for folder in Path('.').iterdir():
        if not folder.is_dir():
            continue
        for file in folder.rglob('*'):
            if not file.name == 'Performance.csv':
                continue

            df = pd.read_csv(Path.joinpath(folder, file.name))

            create_chart(df, folder, Path.joinpath(folder, 'Performance.png'))
