import matplotlib.pyplot as plt

def plot_precision_performance(history):
    train_precision = history.history.get('precision', [])
    val_precision = history.history.get('val_precision', [])
    
    if not train_precision or not val_precision:
        raise ValueError("El historial no contiene datos de precisión. Verifica las métricas usadas en el modelo.")

    epochs = range(1, len(train_precision) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_precision, 'b-o', label='Precisión de Entrenamiento')
    plt.plot(epochs, val_precision, 'r-o', label='Precisión de Validación')
    plt.title('Precisión de Entrenamiento vs Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid()
    plt.show()


