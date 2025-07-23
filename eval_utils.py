import matplotlib.pyplot as plt
import seaborn as sns

def save_confusion_matrix(y_true, y_pred, output_path, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def save_classification_report_csv(y_true, y_pred, output_path):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(output_path)
    print(f"Classification report CSV saved to: {output_path}")
