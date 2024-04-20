import pandas as pd
import sys
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import configurations as cfg
from collections import defaultdict
import os

MODEL_USED = cfg.MODEL_USED

class EmbeddingEvaluator:
    def __init__(self, dataset_name, model_name, embeddings_dataframe):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.dataset_filename = f'../data/{dataset_name}.pkl'
        self.embeddings_filename = f'../data/{dataset_name}_embeddings_{model_name}.pkl'
        self.embeddings_df = embeddings_dataframe
        self.similarity_matrix = self._precompute_similarity_matrix()
        
    def _precompute_similarity_matrix(self):
        embeddings = np.stack(self.embeddings_df['Embedding'].values)
        return cosine_similarity(embeddings)

    def load_embeddings(self):
        df = pd.read_pickle(self.dataset_filename)
        df['Content'] = df['Title'] + ' ' + df['Description']
        df["Issue_id"] = df["Issue_id"].astype('Int64')

        embeddings_df = pd.read_pickle(self.embeddings_filename)
        embeddings_df = pd.merge(embeddings_df, df[['Issue_id', 'Duplicated_issues']], on='Issue_id', how='left')
        return embeddings_df

    # def compute_f1(self, threshold=0.5):
    #     similarity_matrix = cosine_similarity(self.embeddings_df['Embedding'].tolist())
    #     true_positives = false_positives = false_negatives = true_negatives = 0

    #     for i in range(len(self.embeddings_df)):
    #         for j in range(i + 1, len(self.embeddings_df)):
    #             if similarity_matrix[i, j] > 0.98:
    #                 continue
    #             is_duplicate = (self.embeddings_df['Issue_id'].iloc[j] in self.embeddings_df['Duplicated_issues'].iloc[i]) or (self.embeddings_df['Issue_id'].iloc[i] in self.embeddings_df['Duplicated_issues'].iloc[j])
    #             is_above_threshold = similarity_matrix[i, j] >= threshold
    #             if is_duplicate and is_above_threshold:
    #                 true_positives += 1
    #             elif is_duplicate and not is_above_threshold:
    #                 false_negatives += 1
    #             elif not is_duplicate and is_above_threshold:
    #                 false_positives += 1
    #             else:
    #                 true_negatives += 1

    #     precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    #     recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    #     f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    #     return f1, precision, recall, similarity_matrix, true_positives, false_positives, false_negatives, true_negatives



    def compute_f1(self):
        def calculate_metrics(true_positives, false_positives, false_negatives):
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            return f1, precision, recall
        
        thresholds = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]
        f1_scores = []
        precisions = []
        recalls = []
        true_postive_list = []
        false_positive_list = []
        false_negative_list = []
        true_negative_list = []
        
        similarity_matrix = self.similarity_matrix

        issue_duplicates_map = defaultdict(set)
        for issue_id, duplicated_issues in zip(self.embeddings_df['Issue_id'], self.embeddings_df['Duplicated_issues']):
            issue_duplicates_map[issue_id].update(duplicated_issues)


        for threshold in thresholds:
            true_positives = false_positives = false_negatives = true_negatives = 0
            n = len(self.embeddings_df)
            for i in range(n):
                for j in range(i + 1, n):
                    issue_id_i = self.embeddings_df.at[i, 'Issue_id']
                    issue_id_j = self.embeddings_df.at[j, 'Issue_id']
                    is_duplicate = issue_id_j in issue_duplicates_map[issue_id_i] or issue_id_i in issue_duplicates_map[issue_id_j]
                    is_above_threshold = self.similarity_matrix[i, j] >= threshold

                    if is_duplicate and is_above_threshold:
                        true_positives += 1
                    elif is_duplicate and not is_above_threshold:
                        false_negatives += 1
                    elif not is_duplicate and is_above_threshold:
                        false_positives += 1
                    else:
                        true_negatives += 1

            f1, precision, recall = calculate_metrics(true_positives, false_positives, false_negatives)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            true_postive_list.append(true_positives)
            false_positive_list.append(false_positives)
            false_negative_list.append(false_negatives)
            true_negative_list.append(true_negatives)

        # Plotting precision-recall curve
        plt.figure()
        plt.plot(recalls, precisions, marker='o', linestyle='-', color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        # Saving the plot
        metrics_path = f'./metrics/{self.model_name}'
        if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)
        plot_path = os.path.join(metrics_path, 'precision_recall_curve.png')
        plt.savefig(plot_path)
        plt.show()
        
        return f1_scores, precisions, recalls, thresholds, true_postive_list, false_positive_list, false_negative_list, true_negative_list


    def compute_crossEntropy(self):
        similarity_matrix = self.similarity_matrix

        issue_ids = self.embeddings_df['Issue_id'].values
        duplicated_issues = self.embeddings_df['Duplicated_issues'].apply(set).values  # Convert to set for O(1) lookups

        similarity_scores = []
        ground_truth_labels = []

        for i in range(len(issue_ids)):
            for j in range(i + 1, len(issue_ids)):
                if similarity_matrix[i, j] <= 0.98:  # Only process if similarity is <= 0.98
                    is_duplicate = (issue_ids[j] in duplicated_issues[i]) or (issue_ids[i] in duplicated_issues[j])
                    ground_truth_labels.append(is_duplicate)
                    similarity_scores.append(similarity_matrix[i, j])

        # Convert to PyTorch tensors for computation
        similarity_scores_tensor = torch.tensor(similarity_scores)
        ground_truth_labels_tensor = torch.tensor(ground_truth_labels, dtype=torch.float)

        probabilities = torch.sigmoid(similarity_scores_tensor)
        loss = F.binary_cross_entropy(probabilities, ground_truth_labels_tensor)
        return loss

    def compute_roc_auc(self):
        similarity_matrix = self.similarity_matrix

        issue_ids = self.embeddings_df['Issue_id'].values
        duplicated_issues = self.embeddings_df['Duplicated_issues'].apply(set).values

        ground_truth = []
        scores = []

        for i in range(len(issue_ids)):
            for j in range(i + 1, len(issue_ids)):
                is_duplicate = (issue_ids[j] in duplicated_issues[i]) or (issue_ids[i] in duplicated_issues[j])
                ground_truth.append(int(is_duplicate))
                scores.append(similarity_matrix[i, j])

        ground_truth = np.array(ground_truth)
        scores = np.array(scores)
        fpr, tpr, thresholds = roc_curve(ground_truth, scores)
        roc_auc = auc(fpr, tpr)
        
        metrics_path = f'./metrics/{self.model_name}'

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plot_path = os.path.join(metrics_path, 'roc_auc_plot.png')
        plt.savefig(plot_path)  
        plt.show()

        return roc_auc

if __name__ == "__main__":
    # Initialize the evaluator with the dataset and model name
    evaluator = EmbeddingEvaluator('mozilla_firefox', MODEL_USED)
    # Compute F1 score and related metrics
    f1, precision, recall, similarity_matrix, tp, fp, fn, tn = evaluator.compute_f1(threshold=0.622)
    print(f"True Positives: {tp}, False Negatives: {fn}, True Negatives: {tn}, False Positives: {fp}")
    print(f"F1 Score: {f1}, Precision: {precision}, Recall: {recall}")

    # Compute cross-entropy loss
    loss = evaluator.compute_crossEntropy()
    print(f"Cross Entropy Loss: {loss}")

    # Compute ROC-AUC
    roc_auc = evaluator.compute_roc_auc()
    print(f"ROC-AUC: {roc_auc}")
