# Evaporator Modeling

Forecast evaporator temperatures and compressor power for a refrigeration system using machine learning. Supports multi-step (autoregressive) prediction via a Decision Tree model, with an LSTM implementation planned.

---

## Architecture

```
evap_modeling/
├── app.py                          # Streamlit browser UI
├── main.py                         # CLI entry point (train / predict)
├── data/
│   ├── evap_raw_data.csv           # Raw evaporator sensor readings
│   └── comp_and_other_raw_data.csv # Compressor + weather readings
├── src/
│   ├── config.py                   # Project-wide constants and paths
│   ├── data/
│   │   └── loader.py               # CSV loading and preprocessing
│   ├── models/
│   │   ├── evaporator.py           # Evaporator domain object
│   │   ├── compressor.py           # Compressor domain object
│   │   └── system.py               # RefrigerationSystem — aggregates all objects,
│   │                               #   builds feature/target matrices for predictors
│   └── predictors/
│       ├── base_predictor.py       # Abstract interface: fit / predict / save / load
│       ├── decision_tree_predictor.py  # Implemented: multi-output DT with recursive rollout
│       └── lstm_predictor.py       # Placeholder: PyTorch LSTM (not yet implemented)
└── notebooks/
    └── refrigeration_prediction_and_optimization_using_RNN_and_RL.ipynb
```

### Data flow

```
CSV files
  └─► loader.py  (clean, merge, engineer features)
        └─► RefrigerationSystem  (owns Evaporator + Compressor objects)
              └─► build_sequences()  →  (X, y) windowed arrays
                    └─► Predictor.fit()  →  trained model
                              └─► Predictor.predict(window)  →  n-step forecast
```

### Input features (per time step)

For every evaporator: `<id>_temp`, `<id>_temp_setpoint`
Plus system-wide: `dry_bulb_temp`, `wet_bulb_temp`, `system_on`

### Output targets (per time step)

For every evaporator: `<id>_temp`
Plus: `total_compressor_power`

### Prediction strategy

The predictor uses **autoregressive (recursive) multi-step forecasting**:
- One model predicts all outputs at `t+1` given a flattened look-back window of length `seq_length`.
- To forecast `n_steps` ahead, the model loops — feeding predictions back as inputs for the next step.
- Exogenous inputs (setpoints, weather, system_on) hold their last known value during rollout.

### Predictor interface

All predictors implement the same four-method contract from `BasePredictor`:

| Method | Description |
|---|---|
| `fit(system)` | Train on historical data from a `RefrigerationSystem` |
| `predict(X_window)` | Given a `(seq_length, n_features)` window, return `(n_steps, n_outputs)` |
| `save(path)` | Persist trained model to disk |
| `load(path)` | Restore a saved model without retraining |

---

## Setup

Requires Python 3.12–3.14. Uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

Place the two data files in the `data/` directory before running:
```
data/evap_raw_data.csv
data/comp_and_other_raw_data.csv
```

---

## Running via CLI (`main.py`)

### Train a model

```bash
python main.py train --predictor dt --seq-length 6 --n-steps 12 --save models/dt.pkl
```

| Argument | Default | Description |
|---|---|---|
| `--predictor` | `dt` | Which model to use (`dt` = Decision Tree, `lstm` = LSTM) |
| `--seq-length` | `6` | Look-back window in time steps |
| `--n-steps` | `12` | Forecast horizon in time steps |
| `--save` | _(none)_ | Path to save the trained model; omit to train without saving |

### Run predictions from a saved model

```bash
python main.py predict --predictor dt --load models/dt.pkl
```

| Argument | Default | Description |
|---|---|---|
| `--predictor` | `dt` | Must match the model type that was saved |
| `--load` | _(required)_ | Path to a previously saved `.pkl` file |
| `--seq-length` | `6` | Must match the value used at training time |
| `--n-steps` | `12` | Forecast horizon |

Example output:
```
Predicted next 12 steps:
 Step          H01_temp          H02_temp  ...  total_compressor_power
    1            17.500            16.800  ...                 119.200
    2            18.100            17.200  ...                 120.400
  ...
```

---

## Running via Streamlit UI (`app.py`)

```bash
streamlit run app.py
```

Opens a browser UI at `http://localhost:8501` with:

- **Sidebar sliders** — adjust `seq_length`, `n_steps`, and `max_depth` before training
- **Historical Data** — date range picker + evaporator multiselect → temperature and compressor power charts
- **N-Step Forecast** — click "Train & Predict" to train the model and plot the forecast alongside historical context

The model is cached after training; changing a slider and clicking the button again retrains with the new settings.

---

## Notebooks

The original exploration and LSTM prototype live in:

```
notebooks/refrigeration_prediction_and_optimization_using_RNN_and_RL.ipynb
```

The LSTM implementation from cells 9, 11, and 14 of this notebook is the source to port into `src/predictors/lstm_predictor.py`.
