"""
prepare_data.py

把 data.csv 轉成 BERT 八類情緒分類訓練所需的 train / validation JSONL。

原始資料有兩個欄位：

    text,emotion
    你要不要去吃午餐？,平淡語氣

emotion 是中文，訓練時要先換成整數 0 到 7。

這支程式做的事情：

    data.csv
        ↓
    讀取文字，並把情緒文字轉成整數
        ↓
    檢查格式
        ↓
    以 label 做 stratified train / validation split
        ↓
    data/bert/train.jsonl
    data/bert/valid.jsonl

為什麼要先獨立做資料切分？

因為之後 train_bert.py 每次執行時，都直接讀取固定的 train / validation 檔案，
比較不同模型或超參數時，不會因為每次重新亂數切分而改變資料集。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

# csv：
# data.csv 第 1 列是欄位名稱 text,emotion，後面每一列是一筆資料。
import csv

# json：
# 用來把 Python dict 寫成 JSONL。
import json

# Counter：
# 用來統計 label 0 到 7 各有多少筆。
from collections import Counter

# Path：
# 用來處理輸入與輸出路徑。
from pathlib import Path

# train_test_split：
# 用來切 training / validation。
#
# 這裡使用 stratify=labels，
# 讓 train 與 validation 的 8 種情緒比例盡量接近原始資料。
from sklearn.model_selection import train_test_split


# =============================================================================
# 2. 資料設定
# =============================================================================

# 原始資料檔。
INPUT_FILE = Path("data.csv")

# 訓練只能用整數 label。這個對照決定後不要再改，
# 要和 train_bert.py 的 ID2LABEL、LABEL2ID 相同。
EMOTION_TO_ID = {
    "平淡語氣": 0,
    "開心語調": 1,
    "悲傷語調": 2,
    "憤怒語調": 3,
    "驚奇語調": 4,
    "厭惡語調": 5,
    "關切語調": 6,
    "疑問語調": 7,
}

# 輸出資料夾。
OUTPUT_DIR = Path("data/bert")

# Validation 比例。
#
# 0.2 = 20%
#
# 所以大約：
#     80% → training
#     20% → validation
VALID_RATIO = 0.2

# 固定 random seed。
#
# 讓每次執行都得到相同的 train / validation split，
# 方便重現實驗。
SEED = 42


# =============================================================================
# 3. 輔助函式
# =============================================================================

def load_emotion_csv(path: Path):
    """
    讀取 data.csv。

    回傳：
        texts  : list[str]
        labels : list[int]

    emotion 文字會用 EMOTION_TO_ID 轉成 0 到 7。
    """

    texts = []
    labels = []

    # 記錄被略過的空句子行號。
    # 沒有文字就沒有東西可以讓 BERT 學習。
    skipped_empty_rows = []

    # newline="" 是 csv 模組官方建議的開檔方式。
    # utf-8-sig 可以去掉 Excel 偶爾加上的 BOM。
    with path.open("r", encoding="utf-8-sig", newline="") as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"{path} 沒有欄位名稱。")

        fieldnames = set(reader.fieldnames)
        if "text" not in fieldnames or "emotion" not in fieldnames:
            raise ValueError(
                f"{path} 必須有 text 和 emotion 欄位，"
                f"實際為 {reader.fieldnames}。"
            )

        # 第 1 行是欄位名稱，所以資料從第 2 行開始。
        for line_number, row in enumerate(reader, start=2):

            text = (row.get("text") or "").strip()
            emotion = (row.get("emotion") or "").strip()

            if not text:
                skipped_empty_rows.append(line_number)
                continue

            if emotion not in EMOTION_TO_ID:
                raise ValueError(
                    f"{path} 第 {line_number} 行的 emotion 不在對照表："
                    f"{emotion!r}"
                )

            texts.append(text)
            labels.append(EMOTION_TO_ID[emotion])

    if not texts:
        raise RuntimeError(f"{path} 沒有讀到任何有效資料。")

    if skipped_empty_rows:
        print(
            "已略過空句子："
            f"{len(skipped_empty_rows)} 筆；"
            f"原始行號={skipped_empty_rows}"
        )

    return texts, labels


def save_jsonl(texts, labels, path: Path):
    """
    將文字與標籤寫成 JSONL。

    每一行格式：

        {"sentences": "這間飯店很好", "labels": 1}

    為了和原 notebook 的欄位名稱保持一致，
    這裡繼續使用：
        sentences
        labels
    """

    with path.open("w", encoding="utf-8") as file:
        for text, label in zip(texts, labels):

            row = {
                "sentences": text,
                "labels": label,
            }

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_label_distribution(name, labels):
    """
    顯示 0 到 7 的資料量與比例。
    """

    counts = Counter(labels)
    total = len(labels)

    print(f"{name}：{total} 筆")

    for label in sorted(counts):
        count = counts[label]
        ratio = count / total

        print(
            f"  label={label}: "
            f"{count} 筆 "
            f"({ratio:.2%})"
        )


# =============================================================================
# 4. 主流程
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Step 1：確認原始資料存在
    # -------------------------------------------------------------------------

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"找不到原始資料：{INPUT_FILE.resolve()}"
        )

    # -------------------------------------------------------------------------
    # Step 2：讀取 data.csv
    # -------------------------------------------------------------------------

    texts, labels = load_emotion_csv(INPUT_FILE)

    print("=" * 72)
    print_label_distribution("全部資料", labels)

    # -------------------------------------------------------------------------
    # Step 3：切 train / validation
    # -------------------------------------------------------------------------

    # stratify=labels 很重要。
    #
    # 你的資料不是完全 50 / 50，
    # 所以如果只做普通亂數切分，
    # train 與 validation 的 label 比例可能出現額外偏差。
    #
    # stratify 會讓兩邊盡量維持相似的 class distribution。
    (
        train_texts,
        valid_texts,
        train_labels,
        valid_labels,
    ) = train_test_split(
        texts,
        labels,
        test_size=VALID_RATIO,
        random_state=SEED,
        stratify=labels,
    )

    # -------------------------------------------------------------------------
    # Step 4：建立輸出資料夾
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"

    # -------------------------------------------------------------------------
    # Step 5：儲存 JSONL
    # -------------------------------------------------------------------------

    save_jsonl(
        train_texts,
        train_labels,
        train_path,
    )

    save_jsonl(
        valid_texts,
        valid_labels,
        valid_path,
    )

    # -------------------------------------------------------------------------
    # Step 6：顯示結果
    # -------------------------------------------------------------------------

    print_label_distribution(
        "Training",
        train_labels,
    )

    print_label_distribution(
        "Validation",
        valid_labels,
    )

    print(f"Training JSONL：{train_path}")
    print(f"Validation JSONL：{valid_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
