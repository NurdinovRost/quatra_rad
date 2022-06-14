from typing import Union
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
from tqdm import tqdm
import shutil
from incorrect_samples import incorrect_samples


ROOT = Path.cwd()
PATH_TO_DATAFRAME = ROOT / "data" / "raw" / "data.xlsx"
PATH_TO_IMAGES = ROOT / "data" / "raw" / "images"
PATH_TO_PROCESSED_TRAIN = ROOT / "data" / "processed" / "train"
PATH_TO_PROCESSED_TEST = ROOT / "data" / "processed" / "test"


def copy_binary_files(df: pd.DataFrame, src: Path, dest: Path, mode: str):
    """Copy binary files from the DataFrame from src dir to dest dir.

    Args:
        df (pd.DataFrame): Train or test DataFrame.
        src (Path): src the path to the directory with the binary files from where to copy.
        dest (Path): the path to the dest directory, where the binaries will be copied.
        mode (str): description for tqdm. This param can be train or test.
    """
    for i, row in tqdm(df.iterrows(), desc=f"Copy {mode} images"):
        filename = row.file_name
        src_file = src / filename
        dest_file = dest / "images" / filename
        dest_file.write_bytes(src_file.read_bytes())


def label_fix(row: pd.Series) -> str:
    """Return same label if file_name not in dict incorrect_samples.
    Else return fix label from dict incorrect_samples.

    Args:
        row (pd.Series): row of DataFrame

    Returns:
        label (str): name of container
    """
    label = row.container if row.file_name not in incorrect_samples else incorrect_samples[row.file_name]
    return label


def correct_data(df: pd.DataFrame) -> pd.DataFrame:
    """Correcting raw data. Label fix and filtering 'unknown' data.

    Args:
        df (pd.DataFrame): raw DataFrame

    Returns:
        df_fix (pd.DataFrame): Correct DataFrame
    """
    df["container"] = df.apply(lambda row: label_fix(row), axis=1)
    df_fix = df[df.container != "unknown"]

    return df_fix


def create_train_and_test_dirs(path_train: Path, path_test: Path):
    """Creates the final train and test directories. If these directories exist,
    they are deleted and re-created.

    Args:
        path_train: (Path) - Path to processed train directory
        path_test: (Path) - Path to processed test directory
    """
    processed_paths = [path_train, path_test]
    for folder in processed_paths:
        if folder.exists():
            shutil.rmtree(folder)
        folder_images = folder / "images"
        folder_images.mkdir(parents=True)


def split_and_save_dataset(path_to_dataframe: Union[str, Path]):
    """Split DataFrame from interim directory on the test and train DataFrames.
    The training data are 90% of the total data, the test data are 10% of the total data.
    Binary files and data frames themselves are also copiedю

    Args:
        path_to_dataframe (Union[str, Path]): path to interim DataFrame.
    """
    df = pd.read_excel(path_to_dataframe)
    df_fix = correct_data(df)
    df_train, df_test = train_test_split(df_fix, test_size=0.1, random_state=541, shuffle=True)

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    create_train_and_test_dirs(PATH_TO_PROCESSED_TRAIN, PATH_TO_PROCESSED_TEST)
    copy_binary_files(df_train, PATH_TO_IMAGES, PATH_TO_PROCESSED_TRAIN, "train")
    copy_binary_files(df_test, PATH_TO_IMAGES, PATH_TO_PROCESSED_TEST, "test")

    df_train["image_path"] = df_train.apply(lambda row: str(PATH_TO_PROCESSED_TRAIN / "images" / row.file_name), axis=1)
    df_test["image_path"] = df_test.apply(lambda row: str(PATH_TO_PROCESSED_TEST / "images" / row.file_name), axis=1)

    kf = KFold(n_splits=5)
    for fold, (_, val_) in enumerate(kf.split(X=df_train, y=df_train.container)):
        df_train.loc[val_, "fold"] = fold

    df_train.to_csv(PATH_TO_PROCESSED_TRAIN / "data.csv")
    df_test.to_csv(PATH_TO_PROCESSED_TEST / "data.csv")


if __name__ == "__main__":
    split_and_save_dataset(PATH_TO_DATAFRAME)


