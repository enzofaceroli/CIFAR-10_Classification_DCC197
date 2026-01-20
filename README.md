# Classificação de Imagens no CIFAR-10 com CNNs

Este repositório contém o trabalho desenvolvido para a disciplina de Redes Neurais/Visão Computacional. O objetivo foi implementar e analisar o desempenho de diferentes arquiteturas de Redes Neurais Convolucionais (CNNs) na classificação do dataset **CIFAR-10**.

## Arquiteturas Implementadas

Foram testados três modelos principais, utilizando tanto implementação manual quanto *Transfer Learning*:

* **VGG16** (Implementação manual e pré-treinada)
* **ResNet50**
* **DenseNet121**

Os experimentos variaram parâmetros como *Batch Size*, *Learning Rate* e número de épocas.

## Como Executar

Todo o código foi desenvolvido para rodar diretamente no **Google Colab**, não sendo necessária nenhuma instalação local.

Para executar os testes e visualizar os resultados, acesse o notebook através do link abaixo:

👉 **[Abrir Notebook no Google Colab](https://colab.research.google.com/drive/1yqPCl7WWpgcLfMbfmA4Tkg1VdYc3fhYU?usp=sharing)**

### Instruções Rápidas:
1. Clique no link acima.
2. No menu superior do Colab, vá em **Ambiente de execução** > **Alterar o tipo de ambiente de execução** e certifique-se de que a **GPU (T4)** está selecionada.
3. Clique em **Ambiente de execução** > **Executar tudo** (ou rode as células sequencialmente).

## Tecnologias Utilizadas

* Python 3.10
* PyTorch & TorchVision
* Google Colab (GPU NVIDIA Tesla T4)

---
**Autores:** Davi Brito, Enzo Faceroli, Vitor Trindade do Vale