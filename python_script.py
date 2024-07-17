import os
import neptune
import neptune.integrations.prophet as npt_utils
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_roc, plot_precision_recall
import torch
import plotly.express as px

# Initialize the Neptune run
run = neptune.init_run(
    api_token=os.getenv("NEPTUNE_API_TOKEN"),  # get from your repo secret
    project='tlr62/ST-Runs',
    name="awesome-woodpecker",
    tags=["maskRCNN", "finetune"],
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
    
    run["train/epoch/loss"].append(loss)
    run["train/epoch/accuracy"].append(acc)

# Log evaluation results
eval_acc = ...  # Replace with actual evaluation accuracy
eval_loss = ...  # Replace with actual evaluation loss
run["evaluation/accuracy"] = eval_acc
run["evaluation/loss"] = eval_loss

# Log ROC and precision-recall curves
y_test = ...  # Replace with actual test labels
y_pred_proba = ...  # Replace with actual prediction probabilities

fig, ax = plt.subplots()
plot_roc(y_test, y_pred_proba, ax=ax)
run["evaluation/ROC"].upload(fig)

fig, ax = plt.subplots()
plot_precision_recall(y_test, y_pred_proba, ax=ax)
run["evaluation/precision-recall"].upload(fig)

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

# Prophet integration
dataset = pd.read_csv(
    "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
)
model = Prophet()
model.fit(dataset)

# Log Prophet summary
run["prophet_summary"] = npt_utils.create_summary(
    model=model,
    df=dataset,
    log_interactive=True
)

# Get the model configuration
run["model_config"] = npt_utils.get_model_config(model)

# Serialize the model
run["model"] = npt_utils.get_serialized_model(model)

# Get forecast components
predicted = model.predict(dataset)
run["forecast_components"] = npt_utils.get_forecast_components(model, predicted)

# Log forecast plots
run["forecast_plots"] = npt_utils.create_forecast_plots(model, predicted)

# Log residual diagnostics plots
run["residual_diagnostics_plot"] = npt_utils.create_residual_diagnostics_plots(
    predicted, dataset.y
)

# Plotly integration
df_plotly = px.data.iris()
plotly_fig = px.scatter_3d(
    df_plotly, x="sepal_length", y="sepal_width", z="petal_width", color="species"
)
run["interactive_img"].upload(plotly_fig)

# Stop the run
run.stop()
