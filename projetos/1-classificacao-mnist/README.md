# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Vitoria fernanda fonseca pereira

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura da CNN implementada possui 3 blocos convolucionais. Cada bloco é composto por uma camada `Conv2D` com ativação ReLU, seguida de `BatchNormalization` para estabilizar o treinamento e acelerar a convergência, e `MaxPooling2D` para redução de dimensionalidade. Após os blocos convolucionais, os dados são achatados (`Flatten`), e uma camada de `Dropout` (taxa de 0.5) é aplicada para mitigar overfitting. A camada de saída é uma `Dense` com 10 neurônios e ativação `softmax` para classificação das 10 classes do MNIST. Foi utilizado 20% dos dados para validação (`validation_split=0.2`), e o treinamento contou com `EarlyStopping` monitorando a perda de validação (`val_loss`) com paciência de 3 épocas.

### 2️⃣ Bibliotecas Utilizadas

- `tensorflow` (versão 2.12 ou superior)
- `numpy`

### 3️⃣ Técnica de Otimização do Modelo

Foi utilizada a técnica de **Dynamic Range Quantization** (Quantização Dinâmica) nativa do TensorFlow Lite. Essa técnica converte os pesos do modelo de ponto flutuante de 32 bits (`float32`) para inteiros de 8 bits (`int8`), reduzindo drasticamente o tamanho do modelo (em até 4x) e acelerando as operações, enquanto mantém uma precisão muito semelhante à do modelo original.

### 4️⃣ Resultados Obtidos

- **Acurácia de Validação:** 99.22%
- **Tamanho do `model.h5` original:** 1294.88 KB
- **Tamanho do `model.tflite` otimizado:** 113.96 KB (redução de ~91% no tamanho)

### 5️⃣ Comentários Adicionais (Opcional)

A arquitetura simples (3 blocos Conv2D) foi suficiente para obter uma alta precisão (>99%) de forma rápida, adequada para as restrições do desafio. A utilização da quantização pós-treinamento com Dynamic Range Quantization mostrou-se extremamente eficaz na compressão do modelo para cenários de Edge AI, impactando minimamente a precisão final do modelo, evidenciando o grande benefício para sistemas embarcados.

### 6️⃣ Exemplo de Inferência

```text
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```
**Comentário:** O modelo conseguiu classificar corretamente e com alta confiança os 5 primeiros exemplos do conjunto de teste, mostrando a robustez da rede mesmo após a redução drástica do tamanho e precisão dos pesos com a quantização para o TensorFlow Lite.
