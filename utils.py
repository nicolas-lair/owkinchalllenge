import pandas as pd


def save_test_predictions(ids_test, predictions, data_dir):
    """
    Piece of code taken from the baselines for usage in the trainer file
    Save the test predictions in the right format for submission on the platform
    """
    ids_number_test = [i.split("ID_")[1] for i in ids_test]
    test_output = pd.DataFrame({"ID": ids_number_test, "Target": predictions})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv(data_dir / "preds_test_baseline.csv")


def get_params(args):
    """
    Piece of code taken from the baselines for usage in the trainer file
    Get the train and test set + labels + test ids for the submission files
    """
    # -------------------------------------------------------------------------
    # Load the data
    assert args.data_dir.is_dir()

    train_dir = args.data_dir / "train_input" / "resnet_features"
    test_dir = args.data_dir / "test_input" / "resnet_features"

    train_output_filename = args.data_dir / "train_output.csv"

    train_output = pd.read_csv(train_output_filename)

    # Get the filenames for train
    filenames_train = [train_dir / "{}.npy".format(idx) for idx in train_output["ID"]]
    for filename in filenames_train:
        assert filename.is_file(), filename

    # Get the labels
    labels_train = train_output["Target"].values

    assert len(filenames_train) == len(labels_train)

    # Get the numpy filenames for test
    filenames_test = sorted(test_dir.glob("*.npy"))
    for filename in filenames_test:
        assert filename.is_file(), filename
    ids_test = [f.stem for f in filenames_test]
    return filenames_train, filenames_test, labels_train, ids_test