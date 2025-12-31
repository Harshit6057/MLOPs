# import os
# import warnings
# import sys

# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import ElasticNet
# from urllib.parse import urlparse
# import mlflow
# from mlflow.models.signature import infer_signature
# import mlflow.sklearn
# import dagshub
# import logging
# import dagshub
# dagshub.init(repo_owner='khushwantsingh2372005', repo_name='MLflow-and-Dagshub', mlflow=True)


# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)


# def eval_metrics(actual, pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))
#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual, pred)
#     return rmse, mae, r2



# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     np.random.seed(40)

#     # Read the wine-quality csv file from the URL
#     csv_url = (
#         "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
#     )
#     try:
#         data = pd.read_csv(csv_url, sep=";")
#     except Exception as e:
#         logger.exception(
#             "Unable to download training & test CSV, check your internet connection. Error: %s", e
#         )

#     # Split the data into training and test sets. (0.75, 0.25) split.
#     train, test = train_test_split(data)

#     # The predicted column is "quality" which is a scalar from [3, 9]
#     train_x = train.drop(["quality"], axis=1)
#     test_x = test.drop(["quality"], axis=1)
#     train_y = train[["quality"]]
#     test_y = test[["quality"]]


#     alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
#     l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5



#     with mlflow.start_run():
#         lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
#         lr.fit(train_x, train_y)

#         predicted_qualities = lr.predict(test_x)

#         (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

#         print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
#         print("  RMSE: %s" % rmse)
#         print("  MAE: %s" % mae)
#         print("  R2: %s" % r2)

#         mlflow.log_param("alpha", alpha)
#         mlflow.log_param("l1_ratio", l1_ratio)
#         mlflow.log_metric("rmse", rmse)
#         mlflow.log_metric("r2", r2)
#         mlflow.log_metric("mae", mae)

        
#         # For remote server only (Dagshub)
#         remote_server_uri = "https://dagshub.com/khushwantsingh2372005/MLflow-and-Dagshub.mlflow"
#         mlflow.set_tracking_uri(remote_server_uri)



#         tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

#         # Model registry does not work with file store
#         if tracking_url_type_store != "file":
#             # Register the model
#             # There are other ways to use the Model Registry, which depends on the use case,
#             # please refer to the doc for more information:
#             # https://mlflow.org/docs/latest/model-registry.html#api-workflow
#             mlflow.sklearn.log_model(
#                 lr, "model", registered_model_name="ElasticnetWineModel")
#         else:
#             mlflow.sklearn.save_model(lr, "model")
#             mlflow.log_artifacts("model", artifact_path="model")



import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse

import dagshub

# -----------------------------
# DAGSHUB + MLFLOW SETUP
# -----------------------------
dagshub.init(
    repo_owner="khushwantsingh2372005",
    repo_name="MLflow-and-Dagshub",
    mlflow=True
)

remote_server_uri = "https://dagshub.com/khushwantsingh2372005/MLflow-and-Dagshub.mlflow"
mlflow.set_tracking_uri(remote_server_uri)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
np.random.seed(42)

# -----------------------------
# METRIC FUNCTION
# -----------------------------
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    # Load dataset
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    data = pd.read_csv(csv_url, sep=";")

    train, test = train_test_split(data, test_size=0.25, random_state=42)

    train_x = train.drop(columns=["quality"])
    test_x = test.drop(columns=["quality"])
    train_y = train["quality"]
    test_y = test["quality"]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # -----------------------------
    # START MLFLOW RUN
    # -----------------------------
    with mlflow.start_run():

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)

        predictions = model.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predictions)

        # -----------------------------
        # LOG PARAMETERS
        # -----------------------------
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # -----------------------------
        # LOG METRICS (FOR GRAPHS)
        # -----------------------------
        for step in range(5):
            mlflow.log_metric("rmse", rmse, step=step)
            mlflow.log_metric("mae", mae, step=step)
            mlflow.log_metric("r2", r2, step=step)

        # Dataset stats
        mlflow.log_metric("train_rows", train_x.shape[0])
        mlflow.log_metric("test_rows", test_x.shape[0])
        mlflow.log_metric("num_features", train_x.shape[1])

        # -----------------------------
        # PLOT: ACTUAL VS PREDICTED
        # -----------------------------
        plt.figure()
        plt.scatter(test_y, predictions, alpha=0.6)
        plt.xlabel("Actual Quality")
        plt.ylabel("Predicted Quality")
        plt.title("Actual vs Predicted")
        plt.savefig("actual_vs_predicted.png")
        mlflow.log_artifact("actual_vs_predicted.png")
        plt.close()

        # -----------------------------
        # PLOT: RESIDUALS
        # -----------------------------
        residuals = test_y.values - predictions
        plt.figure()
        plt.scatter(predictions, residuals)
        plt.axhline(0, color="red")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.savefig("residual_plot.png")
        mlflow.log_artifact("residual_plot.png")
        plt.close()

        # -----------------------------
        # FEATURE IMPORTANCE
        # -----------------------------
        importance = pd.DataFrame({
            "feature": train_x.columns,
            "coefficient": model.coef_
        })

        importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        plt.figure(figsize=(8, 6))
        plt.barh(importance["feature"], importance["coefficient"])
        plt.title("Feature Importance (ElasticNet)")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        # -----------------------------
        # LOG MODEL WITH SIGNATURE
        # -----------------------------
        signature = infer_signature(train_x, model.predict(train_x))

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="ElasticnetWineModel",
            signature=signature
        )

        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")
