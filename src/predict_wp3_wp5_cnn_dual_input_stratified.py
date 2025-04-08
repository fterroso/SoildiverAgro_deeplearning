from pathlib import Path
import pandas as pd
import modelling
import re
import os


from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping


# Define  training  parameters needed for CNN
EPOCH = 1000  # epochs
B_size = 16  # batch_size

RESULTS_DIRECTORY = Path(".") / "results_strat"
REPORTS_DIRECTORY = Path(".") / "reports_strat"
FIGURES_DIRECTORY = Path(".") / "figures_strat"
DATA_DIRECTORY = Path(".") / "input"

for dir_path in [RESULTS_DIRECTORY, REPORTS_DIRECTORY, FIGURES_DIRECTORY]:
    dir_path.mkdir(parents=True, exist_ok=True)
(RESULTS_DIRECTORY / "errors").mkdir(parents=True, exist_ok=True)
(FIGURES_DIRECTORY / "learning_curves").mkdir(parents=True, exist_ok=True)


climatic_physicochemical_all = pd.read_pickle(DATA_DIRECTORY / "climatic_physicochemical_wp3_wp5.pkl") # red and violet columns


# Drop na:
climatic_physicochemical_all = climatic_physicochemical_all.drop(columns=["CaCO3", "NH4"])

pedoclimatic = pd.read_pickle(DATA_DIRECTORY / "pedoclimatic_wp3_wp5.pkl") # 'pedoclimatic region' column
spectra = pd.read_pickle(DATA_DIRECTORY / "spectra_wp3_wp5.pkl")
cwt_coeffs = np.load(DATA_DIRECTORY / "cwt_coeffs_wp3_wp5.npy")
outputs = pd.read_pickle(DATA_DIRECTORY / "outputs_wp3_wp5.pkl") # blue, green, orange and pink columns

#Dataset preliminary view
print("*"*10, "pedoclimatic","*"*10,)
print(pedoclimatic.info())
print("*"*10, "climatic_physicochemical","*"*10,)
print(climatic_physicochemical_all.info())
print(climatic_physicochemical_all.head())
print("*"*10,"spectra","*"*10)
print(spectra.info())
print("*"*10,"cwt_coeffs","*"*10)
print(cwt_coeffs.shape)
print("*"*10,"outputs","*"*10)
print(outputs.info())

exit(0)

climatic_physicochemical_priority_lst = [
    ["Mean T in ºC (in the previous 12 months to soil sampling)", "Total P in mm (in the previous 12 months to soil sampling)", "pH", "OM", "Nt"],
    ["pHKCl", "EC", "TOC", "POC"],
    ["CEC", "Caex", "Mgex", "Kex", "Naex"],
    ["FMa", "NO3", "Pav"],
    ["Feba", "Cuba", "Znba", "Mnba"],
    ["Agsd_b", "Agsd_c", "Agsd_d", "Agsd_e"]
]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=314159)  ## ver shuffle

# Callback que acumula la historia de entrenamiento entre llamadas a fit()
class CumulativeHistory(Callback):
    def __init__(self, initial_epoch=0):
        super(CumulativeHistory, self).__init__()
        self.cum_epochs = []    # Lista de número de época acumulado
        self.cum_loss = []      # Lista acumulada de training loss
        self.cum_val_loss = []  # Lista acumulada de validation loss
        self.epoch_offset = initial_epoch  # Para continuar la cuenta de época

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # La época global se calcula sumando el offset a la época actual (iniciando en 0)
        global_epoch = self.epoch_offset + epoch + 1
        if not global_epoch in self.cum_epochs:
            self.cum_epochs.append(global_epoch)
            self.cum_loss.append(logs.get("loss"))
            self.cum_val_loss.append(logs.get("val_loss"))

    def update_offset(self):
        # Actualiza el offset según la última época registrada
        if self.cum_epochs:
            self.epoch_offset = self.cum_epochs[-1]
        else:
            self.epoch_offset = 0


def plot_learning_curves(model_name, property_name, cum_history, early_stop_epoch, priority_input):
    """
    Genera un gráfico con las curvas de pérdida de entrenamiento y validación.
    Muestra la época en la que se detuvo el entrenamiento por early stopping.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(cum_history.cum_epochs, cum_history.cum_loss, label="Training Loss")
    plt.plot(cum_history.cum_epochs, cum_history.cum_val_loss, label="Validation Loss")
    
    # Si se pasó una época de early stopping, la marca
    if early_stop_epoch is not None:
        plt.axvline(x=early_stop_epoch, color="r", linestyle="--", 
                    label=f"Early Stopping at epoch {early_stop_epoch}")
    
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss (MSE)", fontsize=16)
    plt.title("Cumulative Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIRECTORY / "learning_curves" / f"learn_curves_{model_name}_{property_name}_input_priority_{priority_input}.pdf", dpi=600)


def fit_model_stratified(X, y, pedoclimatic_values, model, fig, ax, index, label="", **kwargs):
    ## given a model, performs cross validation prediction using stratifiedkfold
    ## prints the summaries of errors
    ## if fig and ax are passed, add to fig and ax points (y_observed, y_pred)
    ## returns a list with errors and Goodness of fit (goof) indicators
    y_pred = cross_val_predict(model, X, y, cv=skf.split(X, pedoclimatic_values))
    errors = y - y_pred
    #print(pd.Series(np.abs(errors).flatten()).describe())
    r2, RMSE, bias, RPIQ, MAPE, MAE, _ = modelling.goof(y.T, y_pred.T)
    if fig is not None:
        modelling.add_goof(y, y_pred, fig, ax, label)
    errors_df = pd.DataFrame(
        {
            "y_observed": y.flatten(),
            "y_pred": y_pred.flatten(),
            "error": errors.flatten(),
        },
        index=index,
    )

    return {"errors": errors_df, "goof": (r2, RMSE, bias, RPIQ,MAPE,MAE)}


def custom_cross_val_predict(deepl_model, X, y, X_shapes, cv, scale=True, **kwargs):
    y_pred = np.zeros_like(y)

    # Instanciar el callback acumulador
    cum_history = CumulativeHistory(initial_epoch=0)
    
    # Extraer callbacks originales de kwargs y agregar el acumulador
    callbacks = kwargs.get("callbacks", [])
    callbacks.append(cum_history)
    kwargs["callbacks"] = callbacks
    # Extraer el número total de epochs planificados para cada fit
    total_epochs = kwargs.get("epochs", 0)

    # cv is expected to have only one element, the loop below has only one iteration
    for train_indexes, test_indexes in cv:
        if scale:
            if len(X_shapes)>1:
                sc = StandardScaler()
                X_train_cnn = sc.fit_transform(X[train_indexes, :X_shapes[0]])
                X_test_cnn = sc.transform(X[test_indexes, :X_shapes[0]])
                
                sc = StandardScaler()
                X_train_mlp = sc.fit_transform(X[train_indexes, X_shapes[0]:])
                X_test_mlp = sc.transform(X[test_indexes, X_shapes[0]:])

                X_train = np.hstack([X_train_cnn, X_train_mlp])
                X_test = np.hstack([X_test_cnn, X_test_mlp])

            else:
                sc = StandardScaler()
                X_train = sc.fit_transform(X[train_indexes, ...])
                X_test = sc.transform(X[test_indexes, ...])
        else:
            X_train = X[train_indexes, ...]
            X_test = X[test_indexes, ...]

        y_train = y[train_indexes]
        ## calibration and validation data
        X_cal, X_val, y_calX, y_valX = train_test_split(
            X_train, y_train, train_size=0.9, random_state=314159
        )
        ## reshape for input to cnn  model.
        X_cal = modelling.construct_X_cnn(X_cal)
        X_val = modelling.construct_X_cnn(X_val)
        X_test = modelling.construct_X_cnn(X_test)

        initial_epoch = cum_history.epoch_offset


        history= None
        if len(X_shapes) > 1: #There are multiple types of inputs
            history= deepl_model.fit(
                [X_cal[:,:X_shapes[0]],X_cal[:,X_shapes[0]:]], 
                y_calX, 
                validation_data=([X_val[:,:X_shapes[0]],X_val[:,X_shapes[0]:]], y_valX), 
                initial_epoch=initial_epoch,
                **kwargs)
            y_pred[test_indexes] = deepl_model.predict([X_test[:,:X_shapes[0]],X_test[:,X_shapes[0]:]])
        else:
            history= deepl_model.fit(
                X_cal, 
                y_calX, 
                validation_data=(X_val, y_valX), 
                initial_epoch=initial_epoch,
                **kwargs)
            y_pred[test_indexes] = deepl_model.predict(X_test)
    cum_history.update_offset()

        
    # Determinar si early stopping se activó
    early_stop_callback = kwargs["callbacks"][1]  # Asume que EarlyStopping es el segundo callback
    
    early_stop_epoch = None
    if early_stop_callback is not None and cum_history.epoch_offset < (len(list(cv)) * total_epochs):
        early_stop_epoch = cum_history.cum_epochs[-1]

    return y_pred, cum_history, early_stop_epoch


def custom_cross_val_predict_w_dual_input(deepl_model, X, y, X_shapes, cv, scale=True, **kwargs):
    y_pred = np.zeros_like(y)

    # Instanciar el callback acumulador
    cum_history = CumulativeHistory(initial_epoch=0)
    
    # Extraer callbacks originales de kwargs y agregar el acumulador
    callbacks = kwargs.get("callbacks", [])
    callbacks.append(cum_history)
    kwargs["callbacks"] = callbacks
    # Extraer el número total de epochs planificados para cada fit
    total_epochs = kwargs.get("epochs", 0)

    # cv is expected to have only one element, the loop below has only one iteration
    for train_indexes, test_indexes in cv:
        X_cnn_train = X[0][train_indexes, ...]
        X_cnn_test = X[0][test_indexes, ...]
        
        if scale:
            sc = StandardScaler()
            X_mlp_train = sc.fit_transform(X[1][train_indexes, ...])
            X_mlp_test = sc.transform(X[1][test_indexes, ...])
        else:
            X_mlp_train = X[1][train_indexes]
            X_mlp_test = X[1][test_indexes]

        y_train = y[train_indexes]

        # calibration and validation data
        cal_indexes, val_indexes = train_test_split(range(X_cnn_train.shape[0]), train_size=0.9, random_state=314159)

        X_cnn_cal= X_cnn_train[cal_indexes,...]
        X_cnn_val= X_cnn_train[val_indexes,...]

        X_mlp_cal= X_mlp_train[cal_indexes,...]
        X_mlp_val= X_mlp_train[val_indexes,...]

        y_cal = y_train[cal_indexes]
        y_val= y_train[val_indexes]

        ## reshape for input to cnn  model.
        X_cnn_cal = modelling.construct_X_cnn(X_cnn_cal)
        X_cnn_val = modelling.construct_X_cnn(X_cnn_val)
        X_cnn_test = modelling.construct_X_cnn(X_cnn_test)

        initial_epoch = cum_history.epoch_offset

    
        history= deepl_model.fit([X_cnn_cal, X_mlp_cal], 
                                 y_cal, 
                                 validation_data=([X_cnn_val,X_mlp_val], y_val), 
                                 initial_epoch=initial_epoch,
                                 **kwargs)
        
        y_pred[test_indexes] = deepl_model.predict([X_cnn_test, X_mlp_test])

    cum_history.update_offset()

        
    # Determinar si early stopping se activó
    early_stop_callback = kwargs["callbacks"][1]  # Asume que EarlyStopping es el segundo callback
    
    early_stop_epoch = None
    if early_stop_callback is not None and cum_history.epoch_offset < (len(list(cv)) * total_epochs):
        early_stop_epoch = cum_history.cum_epochs[-1]

    return y_pred, cum_history, early_stop_epoch


def fit_deepl_model_stratified(X, y, pedoclimatic_values, model, scale, fig, ax, index, X_shapes,  label="", **kwargs):

    if isinstance(X, list): #is dual input
        y_pred, history, early_stop_epoch = custom_cross_val_predict_w_dual_input(
            model,
            X,
            y,
            X_shapes,
            cv= skf.split(X[0], pedoclimatic_values),
            scale=scale,
            **kwargs,
        )
    else:
        y_pred, history, early_stop_epoch = custom_cross_val_predict(
            model,
            X,
            y,
            X_shapes,
            cv= skf.split(X, pedoclimatic_values),
            scale=scale,
            **kwargs,
        )
    errors = y - y_pred
    r2, RMSE, bias, RPIQ, MAPE, MAE, _ = modelling.goof(y.T, y_pred.T)
    if fig is not None:
        modelling.add_goof(y, y_pred, fig, ax, label)
    errors_df = pd.DataFrame(
        {
            "y_observed": y.flatten(),
            "y_pred": y_pred.flatten(),
            "error": errors.flatten(),
        },
        index=index,
    )
    return {"errors": errors_df, "goof": (r2, RMSE, bias, RPIQ, MAPE, MAE), "learning_curves": (history, early_stop_epoch)}


model_pls = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pls", PLSRegression(n_components=20)),
    ]
)
## Parameters of the training and evaluation
early_stop_patience= 30
target_cwt_coeff= 5

property_name_list = outputs.columns.tolist()
num_properties = len(property_name_list)

for priority_input in range(1,len(climatic_physicochemical_priority_lst)+1):
  
    climatic_physicochemical_input_vars= []
    for l in climatic_physicochemical_priority_lst[:priority_input]:
        climatic_physicochemical_input_vars = climatic_physicochemical_input_vars+l 

    print(f"Priority {priority_input}: ","-"*8, climatic_physicochemical_input_vars, "-"*8)
  
    climatic_physicochemical= climatic_physicochemical_all[climatic_physicochemical_input_vars]
    i=1
    for property_name in property_name_list:

        property_name_clean= re.sub(r'[^\w\s]', '', property_name)
        print(f"{i}/{num_properties}","-"*8, property_name, "-"*8)

        report = REPORTS_DIRECTORY / f"report_{property_name_clean}_input_priority_{priority_input}.csv"
        if not os.path.exists(report):
            f_report = open(report, "w")
            f_report.write("model,R2,RMSE,bias,RPIQ,MAPE,MAE\n")
            fig, ax = plt.subplots()

            try:
                print("Fitting PLS...")
                ## first fit a PLS with the same data to compare

                y = outputs.loc[:, property_name].values.reshape(-1, 1)
                X = cwt_coeffs[:, target_cwt_coeff, :]

                # Eliminar filas con valores NaN en y y las mismas en X
                y_nan_mask = ~np.isnan(y).flatten()
                y = y[y_nan_mask]
                X = X[y_nan_mask]
                pedoclimatic_values = pedoclimatic.values[y_nan_mask]
                spectra_filtered = spectra.iloc[y_nan_mask]

                
                fit = fit_model_stratified(
                    X, y, pedoclimatic_values, model_pls, fig, ax, index=spectra_filtered.index, label="PLS using CWT at scale 6"
                )
                r2, RMSE, bias, RPIQ, MAPE, MAE = fit["goof"]
                f_report.write(f"PLS,{r2:.3f},{RMSE:.3f},{bias:.3f},{RPIQ:.3f},{MAPE:.3f},{MAE:.3f}\n")

                fit["errors"].to_csv(
                    RESULTS_DIRECTORY / "errors" / f"errors_{property_name_clean}_PLS_input_priority_{priority_input}.csv"
                )
            except Exception as e:
                print(f"ERROR:: with PLS for property {property_name}", e)
            
            
            # Now the cnn model

            print("Fitting CNNs with single input...")

            for cnn_dim in [1,2]:

                if cnn_dim == 1:
                    print("\tFitting CNN 1D...")
                    model_name = f"CNN1DCWT{target_cwt_coeff+1}"
                    model_label= f"CNN1 using CWT at scale {target_cwt_coeff+1}"

                else:
                    print("\tFitting CNN 2D...")
                    model_name=  "CNN2D"
                    model_label = "CNN2D"
                try:
                    redLR = ReduceLROnPlateau(
                        monitor="val_loss", factor=0.4, patience=30, verbose=1, mode="min"
                    )
                    early_stop = EarlyStopping(
                        monitor="val_loss",  # Supervisa la pérdida en validación
                        patience=early_stop_patience,  # Épocas antes de detener el entrenamiento
                        restore_best_weights=True,  # Restaura los mejores pesos encontrados
                        verbose=1
                    )

                    y = outputs.loc[:, property_name].values.reshape(-1, 1)

                    if cnn_dim == 1:
                        X = cwt_coeffs[:, target_cwt_coeff, :]
                    else:
                        X= cwt_coeffs

                    y_nan_mask = ~np.isnan(y).flatten()
                    y = y[y_nan_mask]
                    X = X[y_nan_mask]
                    pedoclimatic_values = pedoclimatic.values[y_nan_mask]
                    spectra_filtered = spectra.iloc[y_nan_mask]
                    X_shapes = [X.shape]

                    if cnn_dim== 1:
                        cnn = modelling.construct_cnn1d_tutorial_model(X.shape[1])
                    else:
                        cnn= modelling.construct_cnn_model(cwt_coeffs.shape)

                    fit = fit_deepl_model_stratified(
                        X,
                        y,
                        pedoclimatic_values,
                        model=cnn,
                        scale=False,
                        fig=fig,
                        ax=ax,
                        index=spectra_filtered.index,
                        X_shapes = X_shapes,
                        label= model_label,
                        epochs=EPOCH,
                        batch_size=B_size,
                        callbacks=[redLR, early_stop],
                    )
                    r2, RMSE, bias, RPIQ, MAPE, MAE= fit["goof"]
                    f_report.write(f"{model_name},{r2:.3f},{RMSE:.3f},{bias:.3f},{RPIQ:.3f},{MAPE:.3f},{MAE:.3f}\n")

                    fit["errors"].to_csv(
                        RESULTS_DIRECTORY / "errors" / f"errors_{property_name_clean}_{model_name}_input_priority_{priority_input}.csv"
                    )

                    plot_learning_curves(model_name, property_name_clean, fit['learning_curves'][0], fit['learning_curves'][1], priority_input)
                except:
                    print(f"ERROR:: with {model_name} for property{property_name}")

            # Now the cnn model with pedoclimatic

            print("Fitting CNNs with dual input...")

            for cnn_dim in [1,2]:

                if cnn_dim == 1:
                    print("\tFitting CNN 1D...")
                    model_name = f"CNN1DCWT{target_cwt_coeff+1}_DI"
                    model_label= f"CNN1D using CWT at scale {target_cwt_coeff+1} and climatic_physicochemical"

                else:
                    print("\tFitting CNN 2D...")
                    model_name=  "CNN2D_DI"
                    model_label= "CNN2D using CWT and climatic_physicochemical"

                try:
                    redLR = ReduceLROnPlateau(
                        monitor="val_loss", factor=0.4, patience=30, verbose=1, mode="min"
                    )

                    early_stop = EarlyStopping(
                        monitor="val_loss",  # Supervisa la pérdida en validación
                        patience= early_stop_patience,         # Épocas antes de detener el entrenamiento
                        restore_best_weights=True,  # Restaura los mejores pesos encontrados
                        verbose=1
                    )

                    y = outputs.loc[:, property_name].values.reshape(-1, 1)
                    y_nan_mask = ~np.isnan(y).flatten()
                    y = y[y_nan_mask]
                    pedoclimatic_values = pedoclimatic.values[y_nan_mask]
                    spectra_filtered = spectra.iloc[y_nan_mask]

                    if cnn_dim == 1:
                        X = np.hstack([cwt_coeffs[:, target_cwt_coeff, :], climatic_physicochemical])
                        X = X[y_nan_mask]
                        X_shapes = [cwt_coeffs[:, target_cwt_coeff, :].shape[1], climatic_physicochemical.shape[1]]
                        cnn2 = modelling.construct_cnn1d_dual_input_model(X_shapes[0], X_shapes[1])
                    else:
                        cwt_coeffs_masked = cwt_coeffs[y_nan_mask]
                        climatic_physicochemical_masked= climatic_physicochemical.values[y_nan_mask]
                        X = [cwt_coeffs_masked, climatic_physicochemical_masked]
                        X_shapes= [cwt_coeffs_masked.shape, climatic_physicochemical_masked.shape[1]]
                        cnn2= modelling.construct_cnn2d_dual_input_model(X_shapes[0], X_shapes[1])

                    fit = fit_deepl_model_stratified(
                        X,
                        y,
                        pedoclimatic_values,
                        model=cnn2,
                        scale=True,
                        fig=fig,
                        ax=ax,
                        index=spectra_filtered.index,
                        X_shapes = X_shapes,
                        label= model_label,
                        epochs=EPOCH,
                        batch_size=B_size,
                        callbacks=[redLR, early_stop],
                    )
                    r2, RMSE, bias, RPIQ, MAPE, MAE = fit["goof"]
                    f_report.write(f"{model_name},{r2:.3f},{RMSE:.3f},{bias:.3f},{RPIQ:.3f},{MAPE:.3f},{MAE:.3f}\n")
                    plot_learning_curves(model_name, property_name_clean, fit['learning_curves'][0], fit['learning_curves'][1], priority_input)
                    fit["errors"].to_csv(RESULTS_DIRECTORY / "errors" / f"errors_{property_name_clean}_{model_name}_input_priority_{priority_input}.csv")
                except Exception as e:
                    print(f"ERROR:: with {model_name} for property {property_name}", e)

            
            # Now the cnn model with pedoclimatic (PCA)

            print("Fitting CNN dual input with PCA...")

            for cnn_dim in [1,2]:

                if cnn_dim == 1:
                    print("\tFitting CNN 1D...")
                    model_name = f"CNN1DCWT{target_cwt_coeff+1}_DIPCA"
                    model_label= f"CNN1D using CWT at scale {target_cwt_coeff+1} and climatic_physicochemical (PCA)"

                else:
                    print("\tFitting CNN 2D...")
                    model_name=  "CNN2D_DIPCA"
                    model_label= "CNN2D using CWT and climatic_physicochemical (PCA)"

                try:
                    redLR = ReduceLROnPlateau(
                        monitor="val_loss", factor=0.4, patience=30, verbose=1, mode="min"
                    )

                    early_stop = EarlyStopping(
                        monitor="val_loss",  # Supervisa la pérdida en validación
                        patience=early_stop_patience,         # Épocas antes de detener el entrenamiento
                        restore_best_weights=True,  # Restaura los mejores pesos encontrados
                        verbose=1
                    )

                    # Aplicar PCA a climatic_physicochemical
                    n_components = min(5, climatic_physicochemical.shape[1])  # Puedes ajustar el número de componentes
                    scaler = StandardScaler()
                    climatic_physicochemical_scaled = scaler.fit_transform(climatic_physicochemical.values)
                    pca = PCA(n_components=n_components)
                    climatic_physicochemical_pca = pca.fit_transform(climatic_physicochemical_scaled)


                    y = outputs.loc[:, property_name].values.reshape(-1, 1)
                    y_nan_mask = ~np.isnan(y).flatten()
                    y = y[y_nan_mask]

                    pedoclimatic_values = pedoclimatic.values[y_nan_mask]
                    spectra_filtered = spectra.iloc[y_nan_mask]

                    if cnn_dim == 1:
                        X = np.hstack([cwt_coeffs[:, target_cwt_coeff, :], climatic_physicochemical_pca])
                        X = X[y_nan_mask]
                        X_shapes = [cwt_coeffs[:, target_cwt_coeff, :].shape[1], climatic_physicochemical_pca.shape[1]]
                        cnn3 = modelling.construct_cnn1d_dual_input_model(X_shapes[0], X_shapes[1])

                    else:
                        cwt_coeffs_masked = cwt_coeffs[y_nan_mask]
                        climatic_physicochemical_pca_masked= climatic_physicochemical_pca[y_nan_mask]
                        X = [cwt_coeffs_masked, climatic_physicochemical_pca_masked]
                        X_shapes= [cwt_coeffs_masked.shape, climatic_physicochemical_pca_masked.shape[1]]
                        cnn3= modelling.construct_cnn2d_dual_input_model(X_shapes[0], X_shapes[1])
                    
                    fit = fit_deepl_model_stratified(
                        X,
                        y,
                        pedoclimatic_values,
                        model=cnn3,
                        scale=False,
                        fig=fig,
                        ax=ax,
                        index=spectra_filtered.index,
                        X_shapes = X_shapes,
                        label= model_label,
                        epochs=EPOCH,
                        batch_size=B_size,
                        callbacks=[redLR, early_stop],
                    )
                    r2, RMSE, bias, RPIQ, MAPE, MAE = fit["goof"]
                    f_report.write(f"{model_name},{r2:.3f},{RMSE:.3f},{bias:.3f},{RPIQ:.3f},{MAPE:.3f},{MAE:.3f}\n")
                    plot_learning_curves(model_name, property_name_clean, fit['learning_curves'][0], fit['learning_curves'][1], priority_input)
                    fit["errors"].to_csv(RESULTS_DIRECTORY / "errors" / f"errors_{property_name_clean}_{model_name}_input_priority_{priority_input}.csv")

                except:
                    print(f"ERROR:: with {model_name} for property {property_name}")

            # Now the MLP model with pedoclimatic

            print("Fitting MLP dual input...")

            model_label = f"MLPCWT{target_cwt_coeff+1}_DI"

            try:
                redLR = ReduceLROnPlateau(
                    monitor="val_loss", factor=0.4, patience=30, verbose=1, mode="min"
                )

                early_stop = EarlyStopping(
                    monitor="val_loss",  # Supervisa la pérdida en validación
                    patience=early_stop_patience,         # Épocas antes de detener el entrenamiento
                    restore_best_weights=True,  # Restaura los mejores pesos encontrados
                    verbose=1
                )

                y = outputs.loc[:, property_name].values.reshape(-1, 1)
                X = np.hstack([cwt_coeffs[:, target_cwt_coeff, :], climatic_physicochemical])
                y_nan_mask = ~np.isnan(y).flatten()
                y = y[y_nan_mask]
                X = X[y_nan_mask]
                pedoclimatic_values = pedoclimatic.values[y_nan_mask]
                spectra_filtered = spectra.iloc[y_nan_mask]

                X_shapes = [X.shape[1]]

                mlp = modelling.construct_mlp_model(X.shape[1])

                fit = fit_deepl_model_stratified(
                    X,
                    y,
                    pedoclimatic_values,
                    model=mlp,
                    scale=False,
                    fig=fig,
                    ax=ax,
                    index=spectra_filtered.index,
                    X_shapes = X_shapes,
                    label="MLP using CWT at scale 6 and climatic_physicochemical",
                    epochs=EPOCH,
                    batch_size=B_size,
                    callbacks=[redLR, early_stop],
                )
                r2, RMSE, bias, RPIQ, MAPE, MAE = fit["goof"]
                f_report.write(f"{model_label},{r2:.3f},{RMSE:.3f},{bias:.3f},{RPIQ:.3f},{MAPE:.3f},{MAE:.3f}\n")

                plot_learning_curves(model_label, property_name_clean, fit['learning_curves'][0], fit['learning_curves'][1], priority_input)

                fit["errors"].to_csv(RESULTS_DIRECTORY / "errors" / f"errors_{property_name_clean}_{model_label}_input_priority_{priority_input}.csv")
            except:
                print(f"ERROR:: with {model_label} for property {property_name}")       

            f_report.close()   

            ax.legend()
            ax.set_title(f"{property_name}")
            plt.tight_layout()
            fig.savefig(FIGURES_DIRECTORY / f"{property_name_clean}_scatter_input_priority_{priority_input}.pdf", dpi=600)

        else:
            print(f"Property {property_name} already processed for input priority {priority_input}")
        i+=1

