import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)


# Columns needded in inuput_dir for evaluation
TRUE_LABEL_COL = "manual_label"
PRED_LABEL_COL = "interaction_label"
DATASET_COL = "dataset"


# Useful functions
def save_confusion_matrix(y_true, y_pred, labels, output_dir, title, filename, normalize=False):

    cm = confusion_matrix(y_true,y_pred,labels=labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_display)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45,ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted labels (automatic)")
    ax.set_ylabel("True labels (manual)")
    ax.set_title(title)

    # to print values in the cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            if normalize:
                text = f"{cm_display[i, j]:.2f}"
            else:
                text = str(cm_display[i, j])

            ax.text(j, i, text,ha="center", va="center")

    plt.tight_layout()
    path = os.path.join(output_dir,filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return cm


#  Metrics function
def compute_metrics(y_true,  y_pred, labels, dataset_name):

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0))

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0))

    # Per class
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0))

    results = []
    # Accuracy / macro / weighted
    results.append({
        "dataset": dataset_name,
        "label": "ALL",
        "accuracy": accuracy,
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        "support": len(y_true)
    })

    results.append({
        "dataset": dataset_name,
        "label": "MACRO_AVERAGE",
        "accuracy": accuracy,
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        "support": len(y_true)
    })

    results.append({
        "dataset": dataset_name,
        "label": "WEIGHTED_AVERAGE",
        "accuracy": accuracy,
        "precision": precision_weighted,
        "recall": recall_weighted,
        "f1": f1_weighted,
        "support": len(y_true)
    })

    # Per classe
    for i, label in enumerate(labels):
        results.append({
            "dataset": dataset_name,
            "label": label,
            "accuracy": accuracy,
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i]
        })

    return pd.DataFrame(results)

# Confusion matrix
def get_confusions(y_true, y_pred, labels, dataset_name):

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    confusions = []

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j: # diagonal ignored as correct guesses
                continue

            count = cm[i, j]
            if count > 0:
                confusions.append({
                    "dataset": dataset_name,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": count})

    return pd.DataFrame(confusions)



##########


# Classification validation
def classification_validation(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Manual annotated file loading...")
    df = pd.read_csv(input_file)
    print(f"\nTotal number of lines : {len(df)}")
    print("\nAvailable coulumns :")
    print(df.columns.tolist())

    # verification of the required columns
    required_columns = [DATASET_COL, TRUE_LABEL_COL, PRED_LABEL_COL]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the input file: {missing_columns}")

    # label cleaning
    print("\nCleaning the interaction labels...")
    # str conversion
    df[TRUE_LABEL_COL] = df[TRUE_LABEL_COL].astype(str).str.strip()
    df[PRED_LABEL_COL] = df[PRED_LABEL_COL].astype(str).str.strip()
    df[DATASET_COL] = df[DATASET_COL].astype(str).str.strip()

    # replacing Nan values in case (even if should not be put manually by human !)
    invalid_values = ["","nan", "None","NaN","none"]
    for col in [TRUE_LABEL_COL, PRED_LABEL_COL]:
        df.loc[df[col].isin(invalid_values), col] = np.nan

    # verification of the labels
    print("\nManual/True interaction labels :")
    print(df[TRUE_LABEL_COL].value_counts(dropna=False))

    print("\nPredicted interaction labels :")
    print(df[PRED_LABEL_COL].value_counts(dropna=False))

    print("\nDatasets :")
    print(df[DATASET_COL].value_counts())


    # interaction class identification
    true_labels = set(df[TRUE_LABEL_COL].dropna().unique())
    pred_labels = set(df[PRED_LABEL_COL].dropna().unique())
    all_labels = true_labels | pred_labels
    labels = sorted(all_labels)
    print("\nUsed classes for the validation :")
    for i, label in enumerate(labels):
        print(f"{i}: {label}")

    # validation only if predicted label AND true label
    valid_df = df.dropna(subset=[TRUE_LABEL_COL, PRED_LABEL_COL]).copy()
    print(f"Evaluated lines : {len(valid_df)}")
    print(f"Ignored lines : {len(df) - len(valid_df)}")
    if len(valid_df) == 0:
        raise ValueError("No line in the input file has manual_label and interaction_label at the same time.")


    print("\nValidating...")
    y_true = valid_df[TRUE_LABEL_COL]
    y_pred = valid_df[PRED_LABEL_COL]


    # Global confusion matrix
    cm_global = save_confusion_matrix(
        y_true,
        y_pred,
        labels,
        output_dir,
        title="Global confusion matrix",
        filename="confusion_matrix_global.png",
        normalize=False
    )

    print("\nGlobal confusion matrix :")
    print(pd.DataFrame(cm_global, index=labels,columns=labels))

    # Normalized global confusion matrix
    cm_global_norm = save_confusion_matrix(
        y_true,
        y_pred,
        labels,
        output_dir,
        title="normalized global confusion matrix",
        filename="confusion_matrix_global_normalized.png",
        normalize=True
    )


    # Global classification report
    print("\nClassification report global :")

    report_global = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    print(report_global)

    with open(os.path.join(output_dir,"classification_report_global.txt"),"w",encoding="utf-8") as f:
        f.write(report_global)

    # Global metrics
    metrics_global = compute_metrics(y_true, y_pred,labels,"ALL")


    #######
    # Evaluation per dataset
    print("Evaluation per dataset...")
    all_metrics = [metrics_global]

    # datasets = (DATASETS if DATASETS is not None else sorted(valid_df[DATASET_COL].unique()))
    datasets = sorted(valid_df[DATASET_COL].unique())

    for dataset in datasets:
        sub = valid_df[valid_df[DATASET_COL] == dataset].copy()

        if len(sub) == 0:
            print(f"\n{dataset} : no data, so ignored")
            continue

        print(f"DATASET : {dataset}")
        print(f"Nuber of interactions : {len(sub)}")

        y_true_ds = sub[TRUE_LABEL_COL]
        y_pred_ds = sub[PRED_LABEL_COL]

        # distribution
        print("\nManual distribution :")
        print(y_true_ds.value_counts())

        print("\nAutomatic distribution :")
        print(y_pred_ds.value_counts())

        # Confusion matrix
        cm = save_confusion_matrix(
            y_true_ds,
            y_pred_ds,
            labels,
            output_dir,
            title=f"Confusion matrix - {dataset}",
            filename=(f"confusion_matrix_{dataset}.png"),
            normalize=False)

        print("\nConfusion matrix :")
        print(pd.DataFrame(cm,index=labels,columns=labels))

        # Normalized confusion matrix
        save_confusion_matrix(
            y_true_ds,
            y_pred_ds,
            labels,
            output_dir,
            title=(f"Normalized confusion matrix - {dataset}"),
            filename=(f"confusion_matrix_{dataset}_normalized.png"),
            normalize=True
        )

        # Classification report
        report = classification_report(
            y_true_ds,
            y_pred_ds,
            labels=labels,
            zero_division=0
        )

        print("\nClassification report :")
        print(report)

        with open(os.path.join(output_dir,f"classification_report_{dataset}.txt"),"w", encoding="utf-8") as f:
            f.write(report)

        # metrics
        metrics_ds = compute_metrics(y_true_ds,y_pred_ds,labels,dataset)
        all_metrics.append(metrics_ds)


    # Save metrics
    metrics_df = pd.concat(all_metrics,ignore_index=True)
    metrics_path = os.path.join(output_dir,"classification_metrics.csv")
    metrics_df.to_csv(metrics_path,index=False, sep=";")

    print("\nMetrics")
    print(metrics_df.to_string(index=False))


    # Confusion table
    print("\nMost frequent confusions:")
    confusion_tables = []

    # Global
    global_confusions = get_confusions(y_true,y_pred,labels,"ALL")
    confusion_tables.append(global_confusions)

    # Per dataset
    for dataset in datasets:
        sub = valid_df[valid_df[DATASET_COL] == dataset]

        if len(sub) == 0:
            continue

        confusion_tables.append(get_confusions(sub[TRUE_LABEL_COL], sub[PRED_LABEL_COL], labels, dataset))


    confusions_df = pd.concat(confusion_tables,ignore_index=True)
    confusions_df = confusions_df.sort_values(["dataset", "count"], ascending=[True, False])
    confusions_path = os.path.join(output_dir,"classification_confusions.csv")
    confusions_df.to_csv(confusions_path,index=False, sep=";")
    print(confusions_df.to_string(index=False))


    # label distribution
    print("\nLabel distribution per dataset")

    distribution = (valid_df.groupby([DATASET_COL, TRUE_LABEL_COL]).size().reset_index(name="count"))
    distribution_path = os.path.join(output_dir,"manual_label_distribution.csv")
    distribution.to_csv(distribution_path, index=False, sep=";")
    print(distribution)


    # Final sum up
    print("\nFinal sum up of the validation with tests:")

    global_accuracy = accuracy_score(y_true,y_pred)

    _, _, global_f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0))

    _, _, global_f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0))


    print(f"Global accuracy: {global_accuracy:.3f}")
    print(f"F1 macro global: {global_f1_macro:.3f}")
    print(f"F1 weighted global: {global_f1_weighted:.3f}")

    print(f"\nFile generated and saved in{output_dir}:")
    for filename in sorted(os.listdir(output_dir)):
        print(f"  - {filename}")

    print("\nDone.")