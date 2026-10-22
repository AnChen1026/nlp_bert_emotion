# 中文句子情緒分類

微調 `bert-base-chinese`，把一句中文分成 8 種情緒語調。原本課堂範例是二元分類（正面／負面評論），這份作業改成多元分類。

## 訓練資料來源

- [Datasets:Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)

從 `Files and versions` 下載 `data.csv`，共 4159 筆，欄位是 `text`、`emotion`。

## 基礎模型

- [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)

## 安裝套件

訓練在 Google Colab（NVIDIA GPU）上執行。版本號用 `pip list` 查詢：

- torch (待填)
- torchvision (待填)
- torchaudio (待填)
- transformers (待填)
- datasets (待填)
- evaluate (待填)
- accelerate (待填)
- scikit-learn (待填)
- matplotlib (待填)

## 說明

### 情緒類別

`emotion` 欄位是文字，微調時要換成整數。對照表寫在 `prepare_data.py` 的 `EMOTION_TO_ID`，`train_bert.py` 的 `ID2LABEL`、`LABEL2ID` 要跟它一致：

| label | 情緒 | 筆數 |
| --- | --- | --- |
| 0 | 平淡語氣 | 705 |
| 1 | 開心語調 | 592 |
| 2 | 悲傷語調 | 486 |
| 3 | 憤怒語調 | 527 |
| 4 | 驚奇語調 | 499 |
| 5 | 厭惡語調 | 404 |
| 6 | 關切語調 | 560 |
| 7 | 疑問語調 | 386 |

### 從二元分類改成多元分類

- `prepare_data.py`：改讀 `data.csv`，用 `EMOTION_TO_ID` 把情緒文字轉成 0–7，再以 stratify 切成 train 80%／valid 20%（3327／832 筆）。
- `train_bert.py`：`NUM_LABELS` 從 2 改成 8，F1 從 `average="binary"` 改成 `average="weighted"`。
- `infer_bert.py`：預測作業指定的 5 句，輸出格式為 `句子 => 情緒 (score)`，score 取到小數點後第 4 位。

### 訓練設定

| 參數 | 值 |
| --- | --- |
| epochs | 3 |
| batch size | 32（gradient accumulation 2） |
| learning rate | 1e-4（linear，warmup 50 steps） |
| max length | 512 |
| 每 30 steps 驗證一次 | 依 `eval_loss` 保留 best model |
| early stopping | patience 3 |

### 預測結果

（訓練完成後執行 `python infer_bert.py`，把結果貼在這裡）

```
我每天都能跟她一起上學，我好開心！ => ? (0.0000)
最好的朋友要離開臺灣了，以後可能不容易再見面... => ? (0.0000)
我覺得我快不行了... => ? (0.0000)
剛剛收到研究所錄取的通知書！ => ? (0.0000)
今年的冬天好像比較晚來。 => ? (0.0000)
```

## 執行方法

```bash
python prepare_data.py
python train_bert.py
python plot_bert_metrics.py
python infer_bert.py
```

## 檔案結構

```
output/bert_emotion_clf/   微調後的 best model（不含 checkpoint-*）
data.csv                   原始資料
data/bert/                 prepare_data.py 產生的 train.jsonl、valid.jsonl
prepare_data.py            資料前處理、label 轉換
train_bert.py              微調
plot_bert_metrics.py       畫 loss / F1 曲線
infer_bert.py              預測作業指定的 5 句
```

## 成果

（放上訓練過程截圖、`output/bert_emotion_clf/loss_curve.png`、`f1_curve.png`）
