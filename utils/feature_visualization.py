import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_representation(test_embeddings, test_predictions, dimension=3):
    """[summary]

    Args:
        test_embeddings np.array([float]): should contain the deep embeddings. Can be obtained using hooks
        test_predictions np.array([float]): contains the predicted output (from softmax)
        dimension (int, optional): Projection dimension to use for the TSNE algorithm. Defaults to 3.
    """
    tsne = TSNE(dimension, verbose=1)
    tsne_proj = tsne.fit_transform(test_embeddings)
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = 10
    for lab in range(num_categories):
        indices = test_predictions == lab
        ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(
            cmap(lab)).reshape(1, 4), label=lab, alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()
