import pandas as pd
import numpy as np
from os import path
#from imr.constants import DATA_DIRECTORY
#from imr.constants import FIGURES_DIRECTORY
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import MaxPooling2D, MaxPooling1D
from tensorflow.keras.layers import GaussianNoise, BatchNormalization, Activation, concatenate
from tensorflow.keras.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from scipy.stats import iqr


def generate_train_test(n_splits, indexes_wp5, random_state):
    # para wp3
    sample_metadata = pd.read_excel(
        path.join(DATA_DIRECTORY, "WP3_final data PQ.xlsx"), sheet_name="Metadata"
    )
    ## Note: I have removed the data validation procedures in
    ## the excel file to avoid a warning from openpyxl.
    ## we add fake pedomclim for wp5
    climate = sample_metadata["PedoClim (choose from list)"].values
    samples = sample_metadata["Sample ID"].values
    # added to accomodate lack of pedoclim info for wp5
    if indexes_wp5 is not None:
        samples = np.concatenate([samples, indexes_wp5.array])
        from numpy.random import default_rng

        rng = default_rng(random_state)
        climate_values = np.unique(climate)
        climate = np.concatenate(
            [
                climate,
                rng.choice(
                    climate_values,
                    size=len(indexes_wp5),
                ),
            ]
        )
    ##
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_test_samples = [
        (list(samples[train]), list(samples[test]))
        for train, test in skf.split(samples, climate)
    ]
    return train_test_samples


def fit_model(model, X, y_df, train_test_samples, **kwargs):
    y_pred = pd.DataFrame()
    ## esto tiene pinta de estar mal, se cambia el orden de train, con isin.
    for train, test in train_test_samples:
        X_train, y_train, X_test, y_test = (
            X[y_df.index.isin(train), :],
            y_df[y_df.index.isin(train)].values,
            X[y_df.index.isin(test), :],
            y_df[y_df.index.isin(test)].values,
        )
        std_scaler = StandardScaler()
        y_train = std_scaler.fit_transform(y_train)
        y_test = std_scaler.transform(y_test)

        model.fit(X_train, y_train, **kwargs)
        y_test_pred = model.predict(X_test)
        index_y_test = y_df.index[y_df.index.isin(test)]
        y_pred = pd.concat(
            [
                y_pred,
                pd.DataFrame({"prediction": y_test_pred.flatten()}, index=index_y_test),
            ]
        )
        y_pred, _ = y_pred.align(y_df[y_df.index.isin(y_pred.index)], axis=0)
    return y_pred


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def write_summary_y_report(y_df):
    sys.stdout.write(
        "\n"
        + "--------------------------------------------------------------------------------"
    )
    sys.stdout.write(
        "\n"
        + "--------------------------------------------------------------------------------"
    )
    sys.stdout.write(f"\n                             Variable: {y_df.columns[0]}")
    sys.stdout.write("\n" + f"Mean: {np.mean(y_df.iloc[:,0]):.3f}")
    sys.stdout.write("\n Percentiles variable\n")
    sys.stdout.write("   Min  |    Q1   |    Me   |    Q3   |    Max \n")
    sys.stdout.write(
        " | ".join(
            [
                f"{p:7.2f}"
                for p in np.percentile(y_df.iloc[:, 0], np.linspace(0, 100, 5))
            ]
        )
    )
    sys.stdout.write("\n Outliers (below):\n")
    qs = np.percentile(y_df.iloc[:, 0], [25, 75])
    sys.stdout.write(", ".join(y_df.index[y_df.iloc[:, 0] < qs[0] - 1.5 * np.ptp(qs)]))
    sys.stdout.write("\n Outliers (above):\n")
    sys.stdout.write(", ".join(y_df.index[y_df.iloc[:, 0] > qs[0] + 1.5 * np.ptp(qs)]))
    sys.stdout.write(
        "\n"
        + "--------------------------------------------------------------------------------"
    )
    sys.stdout.write(
        "\n"
        + "--------------------------------------------------------------------------------\n\n"
    )


def write_error_report(y, y_pred, label):
    sys.stdout.write(
        "\n"
        + "--------------------------------------------------------------------------------"
    )
    sys.stdout.write("\n" + label)
    sys.stdout.write(
        "\n" + f"Mean absolute error: {mean_absolute_error(y, y_pred):.3f}"
    )
    errors = np.abs(y_pred.flatten() - y)
    sys.stdout.write("\n Percentiles abolute error\n")
    sys.stdout.write("   Min  |    Q1   |    Me   |    Q3   |    Max \n")
    sys.stdout.write(
        " | ".join([f"{p:7.2f}" for p in np.percentile(errors, np.linspace(0, 100, 5))])
    )
    sys.stdout.write("\n" + f"R2 score: {r2_score(y, y_pred):.3f}")
    MSE = mean_squared_error(y, y_pred)
    RMSE = np.sqrt(MSE)
    sys.stdout.write("\n" + f"MSE: {MSE:.3f}, RMSE: {RMSE:.3f}")
    sys.stdout.write(
        "\n"
        + "--------------------------------------------------------------------------------"
    )


def plot_errors(y, y_pred, samples, file_suffix):
    data = pd.DataFrame(
        {"y": y, "y_pred": y_pred, "sample": samples, "error": np.abs(y - y_pred)}
    )
    data = data.sort_values("error")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(np.arange(len(y)), "y", data=data, label="true")
    ax.scatter(np.arange(len(y_pred)), "y_pred", data=data, label="predicted")
    ax.set_xticks(np.arange(len(y)))
    ax.set_xticklabels(data["sample"].to_list())
    ax.set_title(file_suffix)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.savefig(path.join(FIGURES_DIRECTORY, "errors", f"errors_{file_suffix}.png"))
    plt.close("all")


def fit_and_report(model, X, y_df, train_test_samples, meta_label, **kwargs):
    y_pred_df = fit_model(model, X, y_df, train_test_samples, **kwargs)
    y = y_df.loc[y_pred_df.index, :].values.flatten()
    y_pred = y_pred_df.values.flatten()
    write_error_report(y, y_pred, meta_label)
    plot_errors(y, y_pred, y_pred_df.index, meta_label)
    return (y_df.loc[y_pred_df.index, :], y_pred_df, (y - y_pred))


def construct_X_cnn(cwt_matrix):
    return cwt_matrix.reshape(*cwt_matrix.shape, 1).astype("float32")


def construct_cnn_model(features_shape):
    model = Sequential()
    model.add(
        Conv2D(
            30,
            (5, 5),
            input_shape=(features_shape[1], features_shape[2], 1),
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))
    # Compile model
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mse"])
    return model

def construct_cnn2d_dual_input_model(cnn_input_shape, mlp_input_shape):
    # Entrada para la rama CNN
    cnn_input = Input(shape=(cnn_input_shape[1], cnn_input_shape[2], 1))

    # Ramo CNN
    x = Conv2D(30, (5, 5), activation="relu")(cnn_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(15, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(50, activation="relu")(x)

    # Entrada para la rama MLP
    mlp_input = Input(shape=mlp_input_shape)

    # Ramo MLP
    y = Dense(128, activation="relu")(mlp_input)
    y = Dense(50, activation="relu")(y)

    # Fusionamos las salidas de CNN y MLP
    merged = concatenate([x, y])

    # Capa final densa para fusionar las dos ramas
    z = Dense(50, activation="relu")(merged)
    z = Dense(1)(z)

    # Definimos el modelo
    model = Model(inputs=[cnn_input, mlp_input], outputs=z)

    # Compilamos el modelo
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mse"])

    return model

def construct_cnn1d_tutorial_model(nb_features):
    inputShape = (nb_features, 1)
    inputs = Input(shape=inputShape)
    f_s = 6
    x = Conv1D(10, f_s, padding="same", strides=2, name="Conv1", activation="relu")(
        inputs
    )
    x = MaxPooling1D(2)(x)

    x = Conv1D(20, f_s, padding="same", name="Conv2", activation="relu")(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(20, f_s, padding="same", name="Conv3", activation="relu")(x)
    x = MaxPooling1D(2)(x)

    x = Flatten()(x)
    x = Dense(8, activation="relu", name="Dense_all.1")(x)
    x = Dense(1, name="Dense_all.2")(x)
    model = Model(inputs=inputs, outputs=x)
    # compile the model
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mse"])
    return model


def construct_cnn1d_model(nb_features):
    inputShape = (nb_features, 1)
    inputs = Input(shape=inputShape)
    f_s = 6
    # add the layers (can be modified)
    x = GaussianNoise(0.2)(inputs)
    x = Conv1D(
        32,
        f_s,
        padding="same",
        name="Conv1",
        kernel_initializer=initializers.he_normal(seed=0),
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D()(x)

    x = Conv1D(64, f_s, padding="same", name="Conv2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(5)(x)

    x = Conv1D(128, f_s, padding="same", name="Conv3")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(5)(x)

    x = Conv1D(256, f_s, padding="same", name="Conv4")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(5)(x)

    x = Dropout(0.4)(x)
    x = Flatten()(x)

    x = Dense(100, activation="relu", name="Dense_all.1")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(1, name="Dense_all.2")(x)

    model = Model(inputs=inputs, outputs=x)

    # compile the model
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mse"])
    return model


def construct_cnn1d_dual_input_model(nb_features_cnn, nb_features_dense):
    # Entrada 1: Procesada por la CNN 1D
    inputShape1 = (nb_features_cnn, 1)
    input1 = Input(shape=inputShape1, name="Input_CNN")
    f_s = 6

    x1 = GaussianNoise(0.2)(input1)
    x1 = Conv1D(32, f_s, padding="same", name="Conv1", 
                kernel_initializer=initializers.he_normal(seed=0))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = MaxPooling1D()(x1)

    x1 = Conv1D(64, f_s, padding="same", name="Conv2")(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = MaxPooling1D(5)(x1)

    x1 = Conv1D(128, f_s, padding="same", name="Conv3")(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = MaxPooling1D(5)(x1)

    x1 = Conv1D(256, f_s, padding="same", name="Conv4")(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = MaxPooling1D(5)(x1)

    x1 = Dropout(0.4)(x1)
    x1 = Flatten()(x1)

    x1 = Dense(100, activation="relu", name="Dense_CNN.1")(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Dense(50, activation="relu", name="Dense_CNN.2")(x1)  # Cambio salida a 50

    # Entrada 2: Procesada por una red densa
    input2 = Input(shape=(nb_features_dense,), name="Input_Dense")

    x2 = Dense(100, activation="relu", name="Dense_Input.1")(input2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Dense(50, activation="relu", name="Dense_Input.2")(x2)

    # Fusión de las dos representaciones
    merged = concatenate([x1, x2], name="Fusion_Layer")

    # Capa final con salida 1
    output = Dense(1, activation="linear", name="Output")(merged)

    model = Model(inputs=[input1, input2], outputs=output)

    # Compilamos el modelo
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mse"])
    
    return model



# Define functions to evaluate accuracy measures
def add_goof(y_obs, y_pred, fig, ax, label):
    ax.scatter(y_obs, y_pred, label=label)
    ax.set_ylabel("Predicted Values", fontsize=13)
    ax.set_xlabel("Observed Values", fontsize=13)
    x_lim = [np.min(y_obs), np.max(y_obs)]
    ax.set_xlim(x_lim)
    ax.set_ylim(x_lim)
    ax.axline(
        [x_lim[0], x_lim[0]],
        slope=1,
        linestyle="--",
        color="red",
        lw=2,
    )

def construct_mlp_model(nb_features):
    # Definir la entrada
    input_layer = Input(shape=(nb_features,), name="Input_MLP")

    # Primera capa densa con 250 neuronas
    x = Dense(250, name="Dense_1")(input_layer)
    x = BatchNormalization()(x)  # Normalización de lotes
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)  # Regularización

    # Segunda capa densa con 100 neuronas
    x = Dense(100, name="Dense_2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    # Tercera capa densa con 50 neuronas
    x = Dense(50, name="Dense_3")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    # Capa de salida con 1 neurona
    output_layer = Dense(1, activation="linear", name="Output")(x)

    # Construcción del modelo
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compilación
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mse"])

    return model



def goof(observed, predicted):
    cormat = np.corrcoef(observed, predicted)
    corr_xy = cormat[0, 1]
    r2 = corr_xy**2
    RMSE = np.sqrt(np.mean((observed - predicted) ** 2))
    bias = np.mean(predicted) - np.mean(observed)
    mx = np.mean(observed)
    my = np.mean(predicted)
    s2x = np.cov(observed)
    s2y = np.cov(predicted)
    sxy = np.mean((observed - mx) * (predicted - my))
    ccc = 2 * sxy / (s2x + s2y + (mx - my) ** 2)
    RPIQ = iqr(observed) / RMSE

    try:
        MAPE= 100 * np.mean(abs((observed - predicted) / observed), axis=-1) #mean_absolute_percentage_error
    except:
        MAPE= [-1]
    MAE= np.mean(np.abs(observed - predicted))

    return r2, RMSE, bias, RPIQ, MAPE[0], MAE, ccc 
