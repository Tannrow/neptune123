import os
import neptune
import neptune.integrations.prophet as npt_utils
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_roc, plot_precision_recall
import torch
import plotly.express as px
import yfinance as yf

# Initialize the Neptune run
run = neptune.init_run(
    project='tlr62/ST-Runs',
    api_token=os.getenv("NEPTUNE_API_TOKEN"),  # Fetch from environment variable
    name="stock-forecasting",
    tags=["prophet", "stock"],
    source_files=["**/*.py", "config.yaml"],
    dependencies="infer",
    capture_hardware_metrics=False,
)

# Define and log metadata
PARAMS = {
    "batch_size": 64,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "optimizer": "Adam",
}
run["parameters"] = PARAMS

# Additional hyperparameters
parameters = {
    "dense_units": 128,
    "activation": "relu",
    "dropout": 0.23,
    "learning_rate": 0.15,
    "batch_size": 64,
    "n_epochs": 30,
}
run["model/parameters"] = parameters

# Add additional parameters
RANDOM_SEED = 42
run["model/parameters/seed"] = RANDOM_SEED

# Example training loop logging
for epoch in range(parameters["n_epochs"]):
    loss = ...  # Replace with actual loss calculation
    acc = ...  # Replace with actual accuracy calculation
    
    run["train/epoch/loss"].append(str(loss))
    run["train/epoch/accuracy"].append(str(acc))

# Log evaluation results
eval_acc = ...  # Replace with actual evaluation accuracy
eval_loss = ...  # Replace with actual evaluation loss
run["evaluation/accuracy"] = str(eval_acc)
run["evaluation/loss"] = str(eval_loss)

# Log ROC and precision-recall curves
y_test = ...  # Replace with actual test labels
y_pred_proba = ...  # Replace with actual prediction probabilities

# Ensure y_pred_proba is a 2D array
import numpy as np
y_pred_proba = np.array(y_pred_proba)

# Add debugging prints to check the shape and content of y_pred_proba
print("y_pred_proba shape:", y_pred_proba.shape)
print("y_pred_proba content:", y_pred_proba)

if y_pred_proba.ndim == 1:
    y_pred_proba = y_pred_proba.reshape(-1, 1)
elif y_pred_proba.ndim == 0:
    raise ValueError("y_pred_proba is empty or not correctly formed")

fig, ax = plt.subplots()
plot_roc(y_test, y_pred_proba, ax=ax)
plt.savefig('roc_curve.png')
run["evaluation/ROC"].upload('roc_curve.png')

fig, ax = plt.subplots()
plot_precision_recall(y_test, y_pred_proba, ax=ax)
plt.savefig('precision_recall_curve.png')
run["evaluation/precision-recall"].upload('precision_recall_curve.png')

# Log sample predictions
sample_predictions = [...]  # Replace with actual sample predictions
for image, predicted_label, probabilities in sample_predictions:
    description = "\n".join(
        [f"class {label}: {prob}" for label, prob in probabilities]
    )
    run["evaluation/predictions"].append(
        image,
        name=predicted_label,
        description=description,
    )

# Log tabular data as DataFrame
df = pd.DataFrame(
    data={
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_probability": y_pred_proba.max(axis=1),
    }
)
run["evaluation/predictions"].upload(neptune.types.File.as_html(df))

# Upload saved model
torch.save(net.state_dict(), "model.pt")
run["model/saved_model"].upload("model.pt")

# Track dataset artifact
run["dataset/train"].track_files("./datasets/train/images")

# Prophet integration with stock data
# Fetch stock data
stock_symbol = 'AAPL'  # Example: Apple Inc.
stock_data = yf.download(stock_symbol, start='2020-01-01', end='2023-01-01')

# Prepare data for Prophet
df = stock_data.reset_index()[['Date', 'Close']]
df.columns = ['ds', 'y']

# Initialize and train Prophet model
model = Prophet()
model.fit(df)

# Log Prophet summary
run["prophet_summary"] = npt_utils.create_summary(
    model=model,
    df=df,
    log_interactive=True
)

# Get the model configuration
run["model_config"] = npt_utils.get_model_config(model)

# Serialize the model
run["model"] = npt_utils.get_serialized_model(model)

# Get forecast components
predicted = model.predict(df)
run["forecast_components"] = npt_utils.get_forecast_components(model, predicted)

# Log forecast plots
run["forecast_plots"] = npt_utils.create_forecast_plots(model, predicted)

# Log residual diagnostics plots
run["residual_diagnostics_plot"] = npt_utils.create_residual_diagnostics_plots(
    predicted, df['y']
)

# Plotly integration
df_plotly = px.data.iris()
plotly_fig = px.scatter_3d(
    df_plotly, x="sepal_length", y="sepal_width", z="petal_width", color="species"
)
run["interactive_img"].upload(plotly_fig)

# Stop the run
run.stop()
