import glob
import argparse
import csv
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.model_selection import train_test_split

import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

INPUT_PATH = "sdc2023"

WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245
WGS84_SQUARED_FIRST_ECCENTRICITY = 6.69437999013e-3
WGS84_SQUARED_SECOND_ECCENTRICITY = 6.73949674226e-3

HAVERSINE_RADIUS = 6_371_000

SAVE_PATH = "train_models_results.csv"


@dataclass
class ECEF:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def to_numpy(self):
        return np.stack([self.x, self.y, self.z], axis=0)

    @staticmethod
    def from_numpy(pos: np.ndarray):
        x, y, z = [np.squeeze(w) for w in np.split(pos, 3, axis=-1)]
        return ECEF(x=x, y=y, z=z)


@dataclass
class BLH:
    lat: np.ndarray
    lng: np.ndarray
    hgt: np.ndarray = 0


def ECEF_to_BLH(ecef: ECEF) -> BLH:
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to geodetic coordinates (latitude, longitude, height).

    Args:
        ecef (ECEF): ECEF coordinates (x, y, z).

    Returns:
        BLH: Geodetic coordinates (latitude, longitude, height).
    """
    a = WGS84_SEMI_MAJOR_AXIS
    b = WGS84_SEMI_MINOR_AXIS
    e2 = WGS84_SQUARED_FIRST_ECCENTRICITY
    e2_ = WGS84_SQUARED_SECOND_ECCENTRICITY
    x = ecef.x
    y = ecef.y
    z = ecef.z
    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(z * (a / b), r)
    B = np.arctan2(z + (e2_ * b) * np.sin(t) ** 3, r - (e2 * a) * np.cos(t) ** 3)
    L = np.arctan2(y, x)
    n = a / np.sqrt(1 - e2 * np.sin(B) ** 2)
    H = (r / np.cos(B)) - n
    return BLH(lat=B, lng=L, hgt=H)


def haversine_distance(blh_1: BLH, blh_2: BLH) -> np.ndarray:
    """
    Calculate the haversine distance between two sets of points on the Earth's surface.

    Args:
        blh_1 (BLH): Geodetic coordinates of the first point.
        blh_2 (BLH): Geodetic coordinates of the second point.

    Returns:
        np.ndarray: Haversine distance between the two sets of points.
    """
    dlat = blh_2.lat - blh_1.lat
    dlng = blh_2.lng - blh_1.lng
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(blh_1.lat) * np.cos(blh_2.lat) * np.sin(dlng / 2) ** 2
    )
    dist = 2 * HAVERSINE_RADIUS * np.arcsin(np.sqrt(a))
    return dist


def pandas_haversine_distance(df1: pd.DataFrame, df2: pd.DataFrame) -> np.ndarray:
    """
    Calculate the haversine distance between two sets of geodetic coordinates.

    Args:
        df1 (pd.DataFrame): First set of geodetic coordinates.
        df2 (pd.DataFrame): Second set of geodetic coordinates.

    Returns:
        np.ndarray: Haversine distance between the two sets of coordinates.
    """
    blh1 = BLH(
        lat=df1["LatitudeDegrees"].to_numpy(),
        lng=df1["LongitudeDegrees"].to_numpy(),
        hgt=0,
    )
    blh2 = BLH(
        lat=df2["LatitudeDegrees"].to_numpy(),
        lng=df2["LongitudeDegrees"].to_numpy(),
        hgt=0,
    )
    return haversine_distance(blh1, blh2)


def ecef_to_lat_lng(
    tripID: str, gnss_df: pd.DataFrame, UnixTimeMillis: pd.Series | np.ndarray
) -> pd.DataFrame:
    """
    Convert ECEF coordinates to geodetic coordinates (latitude, longitude).

    Args:
        tripID (str): Trip ID.
        gnss_df (pd.DataFrame): GNSS data.
        UnixTimeMillis (pd.Series | np.ndarray): Unix time in milliseconds.

    Returns:
        pd.DataFrame: Geodetic coordinates (latitude, longitude).
    """
    ecef_columns = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    columns = ["utcTimeMillis"] + ecef_columns
    ecef_df = (
        gnss_df.drop_duplicates(subset="utcTimeMillis")[columns]
        .dropna()
        .reset_index(drop=True)
    )
    ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy())
    blh = ECEF_to_BLH(ecef)

    TIME = ecef_df["utcTimeMillis"].to_numpy()
    lat = InterpolatedUnivariateSpline(TIME, blh.lat, ext=3)(UnixTimeMillis)
    lng = InterpolatedUnivariateSpline(TIME, blh.lng, ext=3)(UnixTimeMillis)
    return pd.DataFrame(
        {
            #         'tripId' : tripID,
            "utcTimeMillis": UnixTimeMillis,
            "LatitudeDegrees": np.degrees(lat),
            "LongitudeDegrees": np.degrees(lng),
        }
    )


def calc_score(pred_blh: BLH, gt_blh: BLH) -> float:
    """
    Calculate the score of the predicted trajectory.

    Args:
        pred_blh (BLH): Predicted trajectory.
        gt_blh (BLH): Ground truth trajectory.

    Returns:
        float: Score of the predicted trajectory.
    """
    d = haversine_distance(pred_blh, gt_blh)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])
    return d.mean(), score


def print_comparison(lat, lng, gt_lat, gt_lng):
    for lat_val, lng_val, gt_lat_val, gt_lng_val in zip(lat, lng, gt_lat, gt_lng):
        print(
            f"Pred: ({lat_val:<12.7f}, {lng_val:<12.7f}) Ground Truth: ({gt_lat_val:<12.7f}, {gt_lng_val:<12.7f})"
        )


def print_batch(
    amnt: int,
    lat_arr: np.ndarray,
    lng_arr: np.ndarray,
    gt_lat_arr: np.ndarray,
    gt_lng_arr: np.ndarray,
):
    for batch in range(amnt):
        print(f"Val data {batch}")
        print_comparison(
            lat_arr[batch], lng_arr[batch], gt_lat_arr[batch], gt_lng_arr[batch]
        )


class PositionalEncoding(torch.nn.Module):
    """
    This module injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so that the two can be summed. Here, we use
    sine and cosine functions of different frequencies.

    Args:
        d_model (int): The dimension of the model (i.e., the size of the input embeddings).
        max_len (int): The maximum length of the input sequences for which to precompute positional encodings.

    Attributes:
        pe (torch.Tensor): Precomputed positional encodings.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the model (i.e., the size of the input embeddings).
            max_len (int): The maximum length of the input sequences for which to precompute positional encodings.
        """
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embedding_dim).

        Returns:
            torch.Tensor: Tensor with positional encodings added.
        """
        return x + self.pe[: x.size(0)]


class TransformerEncoder(torch.nn.Module):
    """
    Transformer Encoder, which consists of an input linear layer to upscale the input
    dimension to the model dimension, a positional encoding layer, a stack of Transformer encoder layers, and
    a final fully connected (fc) layer for output transformation.

    Args:
        config (object): A configuration object containing the hyperparameters.

    Attributes:
        config (object): The configuration object.
        upscale (torch.nn.Linear): Linear layer to upscale input dimension to model dimension.
        pos_encoder (PositionalEncoding): Positional encoding layer.
        transformer (torch.nn.TransformerEncoder): Stack of Transformer encoder layers.
        fc (torch.nn.Sequential): Fully connected layers for output transformation.
    """

    def __init__(self, config, num_trainable_params):
        super().__init__()
        self.name = "Transformer"
        self.config = config
        self.upscale = torch.nn.Linear(config.input_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_len)
        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            activation=config.activation,
            batch_first=True,
        )
        layer_trainable_params = sum(
            p.numel() for p in transformer_layer.parameters() if p.requires_grad
        )

        self.fc = torch.nn.Sequential()
        for i, num_neurons in enumerate(config.fc_layers[:-1]):
            self.fc.add_module(
                f"fc_{i}", torch.nn.Linear(num_neurons, config.fc_layers[i + 1])
            )
            if i < len(config.fc_layers) - 1:
                self.fc.add_module(f"relu_{i}", torch.nn.ReLU())
        fc_trainable_params = sum(
            p.numel() for p in self.fc.parameters() if p.requires_grad
        )

        num_layers = int(
            (num_trainable_params - fc_trainable_params) / layer_trainable_params
        )
        if abs(num_trainable_params - num_layers * layer_trainable_params) > abs(
            num_trainable_params - (num_layers + 1) * layer_trainable_params
        ):
            num_layers += 1
        self.transformer = torch.nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers
        )

        assert (
            abs(
                num_trainable_params
                - sum(p.numel() for p in self.parameters() if p.requires_grad)
            )
            < num_trainable_params / 20
        ), f"Number of trainable parameters of transformer is not equal to {num_trainable_params}"

    def forward(self, x):
        x = self.upscale(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x


class LSTMEncoder(torch.nn.Module):
    """
    LSTM Encoder, which consists of an input linear layer to upscale the input
    dimension to the model dimension, a stack of LSTM layers, and a final fully connected (fc) layer for output transformation.

    Args:
        config (object): A configuration object containing the hyperparameters.

    Attributes:
        config (object): The configuration object.
        upscale (torch.nn.Linear): Linear layer to upscale input dimension to model dimension.
        lstm (torch.nn.LSTM): Stack of LSTM layers.
        fc (torch.nn.Sequential): Fully connected layers for output transformation.
    """

    def __init__(self, config, num_trainable_params):
        """
        Initializes the LSTMEncoder module.

        Args:
            config (object): A configuration object containing the hyperparameters for the LSTM Encoder.
        """
        super().__init__()
        self.name = "LSTM"
        self.config = config

        # Linear layer to upscale the input dimension to the model dimension
        self.upscale = torch.nn.Linear(config.input_dim, config.d_model)

        # Fully connected layers for output transformation
        self.fc = torch.nn.Sequential()
        for i, num_neurons in enumerate(config.fc_layers[:-1]):
            self.fc.add_module(
                f"fc_{i}", torch.nn.Linear(num_neurons, config.fc_layers[i + 1])
            )
            if i < len(config.fc_layers) - 1:
                self.fc.add_module(f"relu_{i}", torch.nn.ReLU())
        fc_trainable_params = sum(
            p.numel() for p in self.fc.parameters() if p.requires_grad
        )

        lstm_layer = torch.nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=1,
            batch_first=True,
        )
        layer_trainable_params = sum(
            p.numel() for p in lstm_layer.parameters() if p.requires_grad
        )
        num_layers = int(
            (num_trainable_params - fc_trainable_params) / layer_trainable_params
        )
        if abs(num_trainable_params - num_layers * layer_trainable_params) > abs(
            num_trainable_params - (num_layers + 1) * layer_trainable_params
        ):
            num_layers += 1
        # Stack of LSTM layers
        self.lstm = torch.nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=num_layers,
            batch_first=True,
        )
        assert (
            abs(
                num_trainable_params
                - sum(p.numel() for p in self.parameters() if p.requires_grad)
            )
            < num_trainable_params / 20
        ), f"Number of trainable parameters of LSTM is not equal to {num_trainable_params}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LSTMEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor after passing through the LSTM Encoder.
        """
        # Upscale the input dimension to the model dimension
        x = self.upscale(x)

        # Pass through the stack of LSTM layers
        x, _ = self.lstm(x)

        # Transform the output through fully connected layers
        x = self.fc(x)

        return x


class MLP(torch.nn.Module):
    def __init__(self, config, num_trainable_params):
        super(MLP, self).__init__()
        self.name = "MLP"

        input_dim = config.input_dim
        d_model = config.d_model
        output_dim = config.output_dim

        self.layers = nn.Sequential()
        i = 0
        total_params = 0

        # Initial input layer
        self.layers.add_module(f"fc_{i}", nn.Linear(input_dim, d_model))
        self.layers.add_module(f"relu_{i}", nn.ReLU())
        i += 1

        output_layer = nn.Linear(d_model, output_dim)
        while True:
            # Add hidden layers
            self.layers.add_module(f"fc_{i}", nn.Linear(d_model, d_model))
            self.layers.add_module(f"relu_{i}", nn.ReLU())

            new_model = nn.Sequential(self.layers, output_layer)

            new_model_params = sum(
                p.numel() for p in new_model.parameters() if p.requires_grad
            )

            # Check if adding an output layer would reach num_trainable_params
            if (
                abs(num_trainable_params - (new_model_params))
                < num_trainable_params / 20
            ) or (
                new_model_params > num_trainable_params
            ):  # If second condition is true, it will fail the assert
                self.layers = new_model
                break

            i += 1

        assert (
            abs(
                num_trainable_params
                - sum(p.numel() for p in self.parameters() if p.requires_grad)
            )
            < num_trainable_params / 20
        ), f"Number of trainable parameters of MLP is not equal to {num_trainable_params}"


    def forward(self, x):
        return self.layers(x)


class Config:
    """
    Configuration class to hold the hyperparameters and other settings.

    Args:
        config_dict (dict): A dictionary containing the hyperparameters and other settings.

    Attributes:
        input_dim (int): The dimension of the input features.
        d_model (int): The dimension of the model (i.e., the size of the input embeddings).
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        activation (str): The activation function of intermediate layer, relu or gelu.
        num_layers (int): The number of sub-encoder-layers in the encoder.
        fc_layers (list[int]): The number of neurons in the fully connected layers.
        output_dim (int): The dimension of the output features.
        max_seq_len (int): The maximum sequence length.
        val_split (float): The validation split ratio.
    """

    def __init__(self, config_dict):
        self.input_dim = config_dict.get("input_dim")
        self.d_model = config_dict.get("d_model")
        self.nhead = config_dict.get("nhead")
        self.dim_feedforward = config_dict.get("dim_feedforward")
        self.activation = config_dict.get("activation")
        self.num_layers = config_dict.get("num_layers")
        self.fc_layers = config_dict.get("fc_layers")
        self.output_dim = config_dict.get("output_dim")
        self.max_seq_len = config_dict.get("max_seq_len")
        self.val_split = config_dict.get("val_split")


class GNSSDataset(torch.utils.data.Dataset):
    """
    This class represents a custom dataset for Global Navigation Satellite System (GNSS) data.
    It processes prediction and ground truth dataframes, pads sequences to a maximum length,
    and computes the residuals between predictions and ground truth positions.
    Args:
        pred_dfs (list of pandas.DataFrame): List of dataframes containing prediction data.
        gt_dfs (list of pandas.DataFrame): List of dataframes containing ground truth data.

    Attributes:
        pred_dfs (list of pandas.DataFrame): List of dataframes containing prediction data.
        gt_dfs (list of pandas.DataFrame): List of dataframes containing ground truth data.
        sequences (numpy.ndarray): Numpy array of padded prediction sequences.
        labels (numpy.ndarray): Numpy array of residuals (ground truth - prediction).
    """

    def __init__(self, pred_dfs, gt_dfs):
        """
        Initializes the GNSSDataset.
        Args:
            pred_dfs (list of pandas.DataFrame): List of dataframes containing prediction data.
            gt_dfs (list of pandas.DataFrame): List of dataframes containing ground truth data.
        """
        self.pred_dfs = pred_dfs
        self.gt_dfs = gt_dfs
        self.sequences = []
        self.labels = []

        for pred_df in self.pred_dfs:
            x_np = pred_df[
                [
                    "LatitudeDegrees",
                    "LongitudeDegrees",
                    "IonosphericDelayMeters",
                    "TroposphericDelayMeters",
                ]
            ].to_numpy()
            ## pad to max sequence length
            pad = np.zeros((config.max_seq_len - x_np.shape[0], x_np.shape[1]))
            x_np = np.vstack([x_np, pad])
            # x_np = x_np/180
            self.sequences.append(x_np)

        for gt_df in self.gt_dfs:
            y_np = gt_df[["LatitudeDegrees", "LongitudeDegrees"]].to_numpy()
            ## pad to max sequence length
            pad = np.zeros((config.max_seq_len - y_np.shape[0], y_np.shape[1]))
            y_np = np.vstack([y_np, pad])
            # y_np = y_np/180
            self.labels.append(y_np)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

        self.labels = self.labels - self.sequences[:, :, :2]  # just the residuals

        print("seq and label shapes")
        print(self.sequences.shape)
        print(self.labels.shape)

    def __getitem__(self, i):
        """
        Retrieves the sequence and label at index i.
        Args:
            i (int): Index of the data to retrieve.
        Returns:
            tuple: (sequence, label) at index i.
        """
        return self.sequences[i], self.labels[i]

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        Returns:
            int: Number of sequences in the dataset.
        """
        return self.sequences.shape[0]


def is_converged(mean_dists):
    """
    Check if the last 10 val losses have a standard deviation of less than.
    Args:
        mean_dists (list): List of validation losses.
    Returns:
        bool: True if the validation loss has converged, False otherwise.
    """
    if len(mean_dists) < 10:
        return False
    return np.std(mean_dists[-10:]) < 1.0  # TODO: maybe needs readjusting


def save_results(
    save_path,
    model_type,
    lr,
    num_params,
    training_loss,
    best_loss,
    val_loss,
    test_loss,
    training_time,
    inf_time,
    kaggle_score,
    kaggle_test_score,
    epochs,
):
    """
    Add the training results to a CSV file.
    Args:
        save_path (str): Path to save the CSV file.
        val_losses (list): List of validation losses.

    """
    row = [
        model_type,
        lr,
        num_params,
        training_loss,
        best_loss,
        val_loss,
        test_loss,
        training_time,
        inf_time,
        kaggle_score,
        kaggle_test_score,
        epochs,
    ]
    with open(save_path, "a") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def val_model(model, loader, loss_fn):
    mean_dist = 0
    mean_score = 0
    count = 0
    losses = []
    mean_dists = []
    mean_scores = []
    inf_time = 0.0
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    for features, labels in loader:
        # print(features.shape)
        # print(features[0, :20])
        features = features.to(device)
        labels = labels.to(device)

        starter.record()
        pred = model(features)
        ender.record()
        torch.cuda.synchronize()
        inf_time += starter.elapsed_time(ender) / 1000  # seconds

        loss = loss_fn(pred, labels)
        losses.append(float(loss.cpu()))

        features = features.detach().cpu()  # * 180
        pred = pred.detach().cpu()  # * 180
        labels = labels.detach().cpu()  # * 180
        # print(pred.shape)
        pred_lats = pred[:, :, 0] + features[:, :, 0]
        pred_lngs = pred[:, :, 1] + features[:, :, 1]
        gt_lats = labels[:, :, 0] + features[:, :, 0]
        gt_lns = labels[:, :, 1] + features[:, :, 1]

        # Calculate score according to kaggle, height not necessary for distance
        blh1 = BLH(np.deg2rad(pred_lats), np.deg2rad(pred_lngs), hgt=0)
        blh2 = BLH(np.deg2rad(gt_lats), np.deg2rad(gt_lns), hgt=0)

        mean_dist, mean_score = calc_score(blh1, blh2)
        mean_dists.append(mean_dist)
        mean_scores.append(mean_score)
        count += 1

    return (
        np.array(losses).mean(),
        np.array(mean_dists).mean(),
        np.array(mean_scores).mean(),
        inf_time / count,
    )


def train_model(
    model_type,
    num_trainable_params_list,
    train_loader,
    val_loader,
    test_loader,
    config,
    device,
    epochs,
    lr,
):
    n_eval = 4  # evaluate every n_eval epochs

    PATH = "model.pt"

    best_loss = 99999999  # high number

    for num_trainable_params in num_trainable_params_list:
        if model_type == "LSTM":
            model = LSTMEncoder(config, num_trainable_params)
        elif model_type == "Transformer":
            model = TransformerEncoder(config, num_trainable_params)
        elif model_type == "MLP":
            model = MLP(config, num_trainable_params)
        else:
            raise ValueError(f"Model type {model_type} is not supported.")
        model.to(device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        val_losses = []
        mean_dists = []
        converged = False
        start_time = time.time()
        loss = 0
        for epoch in range(epochs):
            if converged:
                break
            print(f"Epoch {epoch + 1} of {epochs}")
            print(f"Loss/train {loss}")
            # Loop over each batch in the dataset
            for batch in train_loader:
                optimizer.zero_grad()  # If not, the gradients would sum up over each iteration

                # Unpack the data and labels
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)

                # Forward propagate
                outputs = model(features)

                # Backpropagation and gradient descent
                loss = loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()

            # Periodically evaluate our model + log to Tensorboard
            if epoch % n_eval == 0:
                model.eval()
                val_loss, mean_dist, mean_score, _ = val_model(
                    model, val_loader, loss_fn
                )
                val_losses.append(val_loss)
                mean_dists.append(mean_dist)

                if val_loss < best_loss:
                    best_loss = val_loss

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": val_loss,
                        },
                        PATH,
                    )

                print(f"Val mean dist {mean_dist}")
                print(f"Val mean score {mean_score}")
                print(f"Loss/val {val_loss}")

                converged = is_converged(mean_dists)

                # turn on training, evaluate turns off training
                model.train()

                if converged:
                    break

        end_time = time.time()
        # Get test loss and inference time
        model.eval()
        test_loss, mean_dist, mean_test_score, inf_time = val_model(
            model, test_loader, loss_fn
        )

        save_results(
            SAVE_PATH,
            model.name,
            lr,
            num_trainable_params,
            float(loss.cpu()),
            best_loss,
            val_loss,
            test_loss,
            end_time - start_time,
            inf_time,
            mean_score,
            mean_test_score,
            epoch,
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--num_trainable_params",
        type=int,
        nargs="+",  # This allows multiple integers to be passed as a list
        required=True,  # This makes the argument mandatory
        help="List of integers representing the number of trainable parameters",
    )
    args = argparser.parse_args()
    num_trainable_params_list = args.num_trainable_params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.memory._record_memory_history()

    pred_dfs = []
    gt_dfs = []

    for dirname in sorted(glob.glob(f"{INPUT_PATH}/train/*/*")):
        drive, phone = dirname.split("/")[-2:]
        tripID = f"{drive}/{phone}"
        gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv")
        gt_df = pd.read_csv(f"{dirname}/ground_truth.csv")

        info_cols = ["IonosphericDelayMeters", "TroposphericDelayMeters"]
        columns = ["utcTimeMillis"] + info_cols
        info_df = (
            gnss_df.drop_duplicates(subset="utcTimeMillis")[columns]
            .fillna(0)
            .reset_index(drop=True)
        )

        for col in info_cols:
            info_df[col] = info_df[col].fillna(
                (info_df[col].bfill() + info_df[col].ffill()) / 2
            )

        pred_df = ecef_to_lat_lng(tripID, gnss_df, gt_df["UnixTimeMillis"])
        pred_df = pd.merge(pred_df, info_df, on="utcTimeMillis", how="left")
        gt_df = gt_df[["LatitudeDegrees", "LongitudeDegrees"]]
        #     print(pred_df.shape)
        #     print(gt_df.shape)
        pred_dfs.append(pred_df)
        gt_dfs.append(gt_df)

    config_dict = {
        "input_dim": 4,
        "d_model": 8,
        "nhead": 4,
        "dim_feedforward": 128,
        "activation": "relu",
        # "num_layers": 6,
        "output_dim": 2,
        "max_seq_len": 3500,
        "val_split": 0.2,
    }
    # For fc layers divide the d_model by 2 until it reaches the output_dim
    fc_layers = [config_dict["d_model"]]
    while fc_layers[-1] > config_dict["output_dim"]:
        fc_layers.append(fc_layers[-1] // 2)

    fc_layers[-1] = config_dict["output_dim"]
    config_dict["fc_layers"] = fc_layers

    config = Config(config_dict)

    # train test split into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        pred_dfs, gt_dfs, test_size=config.val_split, random_state=2
    )
    # split into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=2
    )

    train_dataset = GNSSDataset(X_train, y_train)
    val_dataset = GNSSDataset(X_val, y_val)
    test_dataset = GNSSDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True)
    # train_model(
    #    "LSTM",
    #    num_trainable_params_list,
    #    train_loader,
    #    val_loader,
    #    test_loader,
    #    config,
    #    device,
    #    epochs=10000,
    #    lr=0.001,
    # )
    # train_model(
    #    "Transformer",
    #    num_trainable_params_list,
    #    train_loader,
    #    val_loader,
    #    test_loader,
    #    config,
    #    device,
    #    epochs=10000,
    #    lr=0.001,
    # )
    train_model(
        "MLP",
        num_trainable_params_list,
        train_loader,
        val_loader,
        test_loader,
        config,
        device,
        epochs=10000,
        lr=0.001,
    )
