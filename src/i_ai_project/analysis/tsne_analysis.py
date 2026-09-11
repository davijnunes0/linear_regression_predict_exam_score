import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def apply_tsne(x_data: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Reduz a dimensionalidade dos dados para 2D usando t-SNE.
    """
    tsne = TSNE(n_components=2, random_state=random_state)
    # O fit_transform pode demorar alguns segundos dependendo do tamanho de x_data
    return tsne.fit_transform(x_data)

def plot_tsne_unlabeled(x_tsne: np.ndarray) -> None:
    """
    Plota o resultado do t-SNE sem diferenciar os pontos (Figura sem rótulos).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(x_tsne[:, 0], x_tsne[:, 1], alpha=0.5, color='gray', s=15)
    
    ax.set_title("Clusterização t-SNE - Sem Rótulos", fontsize=14)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    
    plt.savefig("tsne_sem_rotulos.png", dpi=120)
    plt.show()

def plot_tsne_labeled(x_tsne: np.ndarray, y_target: np.ndarray) -> None:
    """
    Plota o t-SNE colorindo os pontos com base na variável alvo (Figura com rótulos).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # O parâmetro 'c' recebe as notas, criando um mapa de calor (cmap)
    scatter = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_target, cmap='viridis', alpha=0.7, s=15)
    
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Nota do Exame (Exam Score)")
    
    ax.set_title("Clusterização t-SNE - Com Rótulos", fontsize=14)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    
    plt.savefig("tsne_com_rotulos.png", dpi=120)
    plt.show()