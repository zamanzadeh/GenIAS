import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
from sklearn.manifold import TSNE

def ARP_score(test_rep: torch.Tensor, gen_rep: torch.Tensor) -> float:
    """
    Compute the Anomalous Representation Proximity (ARP) score between real and generated 
    anomalies in the representation space. A lower average distance between real and 
    generated anomalies results in a higher ARP score.

    The ARP score is computed as:
    ARP = 1 / (1 + avg_min_distance)

    Args:
        test_rep (torch.Tensor): Tensor of real anomaly representations of shape (num_test_samples, dims).
        gen_rep (torch.Tensor): Tensor of generated anomaly representations of shape (num_generated_samples, dims).

    Returns:
        float: The computed ARP score, where a higher value indicates better alignment 
               between real and generated anomalies.
    """

    # Compute pairwise Euclidean distances between test and generated anomaly representations
    distances = torch.cdist(test_rep, gen_rep, p=2)

    # Find the minimum distance for each real anomaly to any generated anomaly
    min_distances, _ = torch.min(distances, dim=1)

    # Compute the ARP score
    arp_score = min_distances.mean().item()
    arp_score = 1.0 / (1.0 + arp_score)

    return arp_score


def get_ARP_scores(all_anomalous: dict, test_representations: torch.Tensor, test_labels: torch.Tensor) -> dict:
    """
    Compute the ARP scores for different anomaly injection methods by comparing 
    the injected anomaly representations against real anomaly representations.

    Args:
        all_anomalous (dict): Dictionary containing injected representations for various anomaly injectors.
                              Expected format:
                              {
                                  "injector_name": {
                                      "injected_representations": torch.Tensor of shape (num_samples, dims)
                                  },
                                  ...
                              }
        test_representations (torch.Tensor): Tensor containing representations of test anomalies of shape (num_samples, dims).
        test_labels (torch.Tensor): Binary tensor indicating anomaly presence (1 for anomaly, 0 for normal).

    Returns:
        dict: A dictionary mapping each injector name to its computed ARP score.
              Format:
              {
                  "injector_name_rs": float,
                  ...
              }
    """

    results = {}
    # Select representations corresponding to actual anomalies
    filtered_representations = test_representations[test_labels == 1]

    for injector in all_anomalous:
        rs = ARP_score(filtered_representations, all_anomalous[injector]['injected_representations'])
        results[f'{injector}_rs'] = rs

    return results


def get_EDI_scores(all_anomalous: dict, k: int) -> dict:
    """
    Compute the Entropy Diversity Index (EDI) score for anomaly representations generated 
    by different injection methods. This measures how diverse the generated anomalies are 
    in terms of their distribution across different partitions of the representation space.

    The process involves:
    1. Sampling a subset of anomaly representations from each injector.
    2. Applying t-SNE to reduce dimensionality.
    3. Normalizing the transformed data and binarizing features.
    4. Assigning binary labels to partitions based on binarized feature values.
    5. Computing Shannon entropy over the partition distribution for each injection method.

    Args:
        all_anomalous (dict): Dictionary containing injected representations for various anomaly injectors.
                              Expected format:
                              {
                                  "injector_name": {
                                      "injected_representations": torch.Tensor of shape (num_samples, dims)
                                  },
                                  ...
                              }
        k (int): Number of components (features) for t-SNE transformation.

    Returns:
        dict: A dictionary mapping each injector name to its computed diversity score.
              Format:
              {
                  "injector_name": float,
                  ...
              }
    """

    datasets = [all_anomalous[injector]['injected_representations'] for injector in all_anomalous]

    np.random.seed(42)
    max_len = datasets[0].shape[0]
    n_samples = int(max_len * 0.1)  # Sample 10% of the dataset for diversity analysis
    indices = np.random.choice(max_len, size=n_samples, replace=False)

    # Sample a subset from each dataset
    sampled_tensors = [tensor[indices] for tensor in datasets]

    # Combine all sampled anomaly representations and apply t-SNE
    combined_data = torch.cat(sampled_tensors, dim=0).cpu().numpy()
    tsne = TSNE(n_components=k, random_state=42)
    combined_data = tsne.fit_transform(combined_data)

    # Normalize feature values
    mins = np.min(combined_data, axis=0)
    maxs = np.max(combined_data, axis=0)
    ranges = np.where(maxs - mins == 0, 1, maxs - mins)  # Avoid division by zero
    norm_samples = (combined_data - mins) / ranges

    # Convert to binary representation (0 or 1 based on threshold 0.5)
    bin_indices = (norm_samples >= 0.5).astype(np.int32)

    # Assign binary partition labels
    powers = 2 ** np.arange((k - 1), -1, -1)  # Binary encoding (e.g., [128, 64, 32, 16, 8, 4, 2, 1])
    labels = bin_indices.dot(powers)

    # Compute overall partition statistics
    unique_partitions, partition_counts = np.unique(labels, return_counts=True)
    probabilities = partition_counts / labels.size

    # Compute diversity scores for each dataset
    diversity_scores = {}
    start_idx = 0
    dataset_names = ["genias_ds", "couta_ds", "ncad_ds", "cutaddpaste_ds", "carla_ds"]

    for dataset_name, dataset in zip(dataset_names, sampled_tensors):
        num_samples = dataset.shape[0]
        end_idx = start_idx + num_samples

        # Compute per-model partition proportions
        dataset_partitions = labels[start_idx:end_idx]
        unique_partitions, partition_counts = np.unique(dataset_partitions, axis=0, return_counts=True)
        dataset_proportions = partition_counts / num_samples

        # Compute Shannon entropy for this dataset
        diversity_scores[dataset_name] = entropy(dataset_proportions, base=2) / float(k)

        start_idx = end_idx

    return diversity_scores
