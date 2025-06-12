# GAT実装実験プロジェクト

このプロジェクトは、Graph Attention Networks (GAT)の様々な設定を実験し、性能への影響を調査するものです。

## プロジェクト構造

```
GAT_experiments/
├── README.md                    # このファイル
├── original/                    # オリジナル実装
│   └── GAT_ipynb_初期.ipynb     # 初期のGAT実装
├── 工夫1/                       # 第1回実験
│   ├── 工夫1.ipynb             # Multi-head数、Dropout、バイアス設定の実験
│   └── 変更点説明.md            # 工夫1の詳細説明
├── 工夫2/                       # 第2回実験
│   ├── 工夫2.ipynb             # GCNConvによる実装変更
│   └── 変更点説明.md            # 工夫2の詳細説明
├── 工夫3/                       # 第3回実験
│   ├── 工夫3.ipynb             # GINConvによる実装変更
│   └── 変更点説明.md            # 工夫3の詳細説明
└── data/                        # データファイル（必要に応じて）
    ├── cora.content
    └── cora.cites
```

## 基本設定（全実験共通）

### データセット
- **Cora dataset**: 学術論文の引用ネットワーク
- **ノード数**: 2708
- **特徴量次元**: 1433
- **クラス数**: 7
- **エッジ数**: 5429

### データ分割（初期実装と同一）
- **訓練データ**: 各クラス20サンプル（計140サンプル）
- **検証データ**: 500サンプル
- **テストデータ**: 1000サンプル

### 共通パラメータ
- **学習率**: 0.0001
- **重み減衰**: 0.0001
- **最大エポック数**: 300
- **最適化器**: Adam
- **Early Stopping**: 有効（patience=20）

### 実験別設定
- **工夫1（GAT）**: 隠れ層32次元、ELU・LeakyReLU、Multi-head Attention
- **工夫2（GCN）**: 隠れ層64次元、ReLU、BatchNorm、cached=True
- **工夫3（GIN）**: 隠れ層64次元、MLP変換、BatchNorm、理論的最強表現力

## 実験一覧

### 工夫1: 基本パラメータの調整実験
- **Multi-head Attention数**: 4, 8, 16の比較
- **Dropout率**: 0.0, 0.3, 0.6の比較
- **バイアス項**: 有無の比較
- **Early Stopping**: 改良された実装

### 工夫2: GATからGCNConvへの変更実験
**目的**: Attention機構とGraph Convolution機構の性能比較

**変更内容**:
- **アーキテクチャ**: GAT → 2層GCNConv
- **正則化強化**: BatchNorm + ReLU + Dropout(0.5)
- **効率化**: cached=True, エッジインデックス形式
- **パラメータ削減**: より少ないパラメータで同等以上の性能を目指す

**技術的特徴**:
- 隠れ層次元: 64
- 活性化関数: ReLU
- PyTorch Geometric使用
- メモリ効率的な疎行列表現

### 工夫3: GCNからGINConvへの変更実験
**目的**: 理論的に最強の表現力を持つGNNの性能検証

**変更内容**:
- **アーキテクチャ**: GCN → 2層GINConv + MLP
- **MLP構成**: [Linear→ReLU→Linear→BatchNorm] × 2層
- **理論的保証**: Weisfeiler-Lehman test同等の表現力
- **表現力向上**: 複雑な非線形変換による特徴学習

**技術的特徴**:
- 隠れ層次元: 64
- MLP: 各GINConvに4層構成
- Dropout: 0.5
- 理論的に最強のGNN表現力

## 実行方法

### 前提条件
```bash
# 基本ライブラリ（全実験共通）
pip install torch matplotlib scikit-learn numpy scipy

# PyTorch Geometric（工夫2・工夫3で必要）
pip install torch-geometric
```

### 実行手順
1. 必要なデータファイル（cora.content, cora.cites）を適切な場所に配置
2. 各実験フォルダ内のJupyterノートブックを実行
3. 結果の比較・分析

### 実験の実行順序（推奨）
1. **工夫1**: GATの基本パラメータ調整実験
2. **工夫2**: GCNConvによる効率的実装
3. **工夫3**: GINConvによる最強表現力実験

## 3手法の比較分析

| 項目 | GAT (工夫1) | GCN (工夫2) | GIN (工夫3) |
|------|-------------|-------------|-------------|
| **主要機構** | Multi-head Attention | Graph Convolution | Graph Isomorphism + MLP |
| **隠れ次元** | 32×8=256 | 64 | 64 |
| **理論的表現力** | 高い | 中程度 | 最強（WL-test同等） |
| **パラメータ数** | 多い | 少ない | 多い |
| **計算効率** | 中程度 | 高い | 低い |
| **学習安定性** | Dropout | BatchNorm | BatchNorm |

## 注意事項

- 全ての実験で**乱数シード（42）**を固定して再現性を確保
- データの前処理・分割は**初期実装と完全に同一**
- GPU利用可能な場合は自動的にGPUを使用
- **工夫2・工夫3**: PyTorch Geometricが必要
- **メモリ使用量**: GAT > GIN > GCN の順

## 期待される実験結果

### 性能面
- **理論的表現力順**: GIN > GAT ≥ GCN
- **実際の精度**: データセットや問題設定による
- **学習安定性**: GIN ≈ GCN > GAT（BatchNormの効果）

### 効率面  
- **学習速度**: GCN > GAT > GIN
- **メモリ効率**: GCN > GIN > GAT
- **実装複雑さ**: GCN < GIN < GAT

この実験により、**Attention**、**Convolution**、**Isomorphism** という3つの異なるアプローチの特性を直接比較し、グラフニューラルネットワークの理解を深めることができます。 