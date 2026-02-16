import os
import pandas as pd
import pytesseract
from PIL import Image
from transformers import LayoutLMv3Processor

# 🔹 Processor (NO apply_ocr)
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False
)

LABELS = [
    "O",
    "B-VALUE","I-VALUE",
    "B-START_DATE","I-START_DATE",
    "B-END_DATE","I-END_DATE",
    "B-RENEWAL","I-RENEWAL",
    "B-PARTY1","I-PARTY1",
    "B-PARTY2","I-PARTY2"
]

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


# ✅ Normalize bbox to 0–1000
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]


# ✅ OCR with normalized boxes
def ocr_with_boxes(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words = []
    boxes = []

    for i, word in enumerate(data["text"]):
        if word.strip() == "":
            continue

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        bbox = normalize_bbox([x, y, x + w, y + h], width, height)

        words.append(word)
        boxes.append(bbox)

    return image, words, boxes


# ✅ Token level labeling (multi-token support)
def label_tokens(words, value, tag):
    labels = ["O"] * len(words)

    if pd.isna(value):
        return labels

    value_tokens = str(value).lower().split()

    for i in range(len(words)):
        if words[i].lower() == value_tokens[0]:
            match = True

            for j in range(len(value_tokens)):
                if i + j >= len(words) or words[i + j].lower() != value_tokens[j]:
                    match = False
                    break

            if match:
                labels[i] = f"B-{tag}"
                for j in range(1, len(value_tokens)):
                    labels[i + j] = f"I-{tag}"

    return labels


# ✅ Create training example
def create_example(image_path, row):

    image, words, boxes = ocr_with_boxes(image_path)

    labels = ["O"] * len(words)

    fields = {
        "VALUE": row.get("Agreement Value"),
        "START_DATE": row.get("Agreement Start Date"),
        "END_DATE": row.get("Agreement End Date"),
        "RENEWAL": row.get("Renewal Notice (Days)"),
        "PARTY1": row.get("Party One"),
        "PARTY2": row.get("Party Two"),
    }

    for tag, value in fields.items():
        token_labels = label_tokens(words, value, tag)

        for i in range(len(labels)):
            if token_labels[i] != "O":
                labels[i] = token_labels[i]

    encoding = processor(
        image,
        words,
        boxes=boxes,
        word_labels=[label2id[l] for l in labels],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    return {k: v.squeeze() for k, v in encoding.items()}


# ✅ Load dataset
def load_dataset(csv_path, image_root_folder):
    df = pd.read_csv(csv_path)
    dataset = []

    for _, row in df.iterrows():

        file_name = str(row["File Name"]).strip()
        folder = os.path.join(image_root_folder, file_name)

        if not os.path.exists(folder):
            print("⚠ Missing folder:", folder)
            continue

        for img in os.listdir(folder):
            if img.endswith(".png"):

                image_path = os.path.join(folder, img)

                try:
                    example = create_example(image_path, row)
                    dataset.append(example)
                except Exception as e:
                    print("❌ Error:", image_path, e)

    print("✅ Loaded samples:", len(dataset))
    return dataset
