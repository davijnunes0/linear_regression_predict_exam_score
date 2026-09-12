from i_ai_project.utils.data_prep import prepare_data
from i_ai_project.analysis.tsne_analysis import apply_tsne, plot_tsne_unlabeled, plot_tsne_labeled

def main_tsne():
    print("1. Carregando e preparando os dados...")
    X_train, X_test, y_train, y_test = prepare_data()

    print("2. Aplicando o t-SNE nos dados de treino (Isso pode levar de 10 a 30 segundos)...")
    # O t-SNE processa melhor em arrays puramente numéricos (já garantido pelo prepare_data)
    X_tsne = apply_tsne(X_train)

    print("3. Gerando figura sem rótulos...")
    plot_tsne_unlabeled(X_tsne)

    print("4. Gerando figura com rótulos (Exam_Score)...")
    plot_tsne_labeled(X_tsne, y_train)
    
    print("Tudo concluído! Gráficos salvos na pasta figures/.")

if __name__ == "__main__":
    main_tsne()