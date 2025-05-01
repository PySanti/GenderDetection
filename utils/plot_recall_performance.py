import matplotlib.pyplot as plt

def plot_recall_performance(history):
    """
    Grafica el recall de entrenamiento (train) y validación (val) del historial del modelo.

    :param history: El historial retornado por el método fit del modelo.
    """
    train_recall = history.history.get('recall', [])
    val_recall = history.history.get('val_recall', [])
    
    if not train_recall or not val_recall:
        raise ValueError("El historial no contiene datos de recall. Verifica las métricas usadas en el modelo.")

    epochs = range(1, len(train_recall) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_recall, 'b-o', label='Recall de Entrenamiento')
    plt.plot(epochs, val_recall, 'r-o', label='Recall de Validación')
    plt.title('Recall de Entrenamiento vs Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid()
    plt.show()

