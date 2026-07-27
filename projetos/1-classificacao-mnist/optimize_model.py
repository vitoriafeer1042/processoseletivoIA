import os
import tensorflow as tf

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.h5")
    tflite_path = os.path.join(script_dir, "model.tflite")
    
    print(f"Carregando modelo de {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Convertendo modelo para TFLite com otimização (Dynamic Range Quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"Modelo TFLite salvo em {tflite_path}")
    print(f"Tamanho do modelo original: {os.path.getsize(model_path) / 1024:.2f} KB")
    print(f"Tamanho do modelo TFLite: {os.path.getsize(tflite_path) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
