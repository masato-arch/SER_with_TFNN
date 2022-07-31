# SER_with_TFNN
大学院で研究してる音声感情認識のプロジェクトを管理するためのリポジトリです．

## 概要
本研究では，Tensor Factorized Neural Network(TFNN)を用いた音声感情認識の有効性を確認する．音声感情認識とは，人の発話音声に含まれる情報のうち非言語の情報，すなわち声の特性(声の大きさや高さ，喋る速さなど)を利用して発話者の感情を推定する技術であり，対話ロボットやメンタルケア，対話のコーチングなど，人と機械のコニュニケーションを要する分野への応用が盛んである．近年は音声をスペクトログラム化してCNNで処理する手法が盛んであり，多くの研究で有効性が示されている．しかしCNNに用いられている全結合層は，画像であるスペクトログラムの空間的なピクセル配置をベクトル化することによって壊してしまうという問題がある．これを解決するのがTFNNである．TFNNはベクトル(1階テンソル)を扱う全結合ネットワークを高階テンソルを扱えるよう一般化したものと考えられる．[Pandey, 2021](https://doi.org/10.1016/j.bspc.2021.103173)ではTFNNがCNN+LSTMで与えられたSoTAを上回る精度を見せた．本研究でもこれを参考にしたアーキテクチャを用いて更に改良をすすめる(まだ実装途中)．

## イントロダクション
Coming soon.

## Tensor Factorized Neural Network
Coming soon.

# 現在の取り組み

Tensor Factorized Neural Network (TFNN)を用いた音声感情認識モデル
> S. K. Pandey, H. S. Shekhawat and S. Shekhawat, "Attention gated tensor neural network architectures for speech emotion recognition," Biomedical Signal Processing and Control, Volume 71, Part A, 2022, 103173, ISSN 1746-8094, https://doi.org/10.1016/j.bspc.2021.103173.

の実装と検証
