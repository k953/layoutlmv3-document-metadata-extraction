import os
import re
import torch
from transformers import LayoutLMv3ForTokenClassification
from dataset import ocr_with_boxes, processor, id2label

# 🔹 Load trained model
model = LayoutLMv3ForTokenClassification.from_pretrained("./model")
model.eval()


# 🔹 Decode BIO predictions → field text
def decode_predictions(words, preds):
    fields = {
        "VALUE": [],
        "START_DATE": [],
        "END_DATE": [],
        "RENEWAL": [],
        "PARTY1": [],
        "PARTY2": []
    }

    for word, p in zip(words, preds):
        label = id2label[p]

        if label.startswith("B-") or label.startswith("I-"):
            tag = label.split("-")[1]
            if tag in fields:
                fields[tag].append(word)

    # join tokens
    for k in fields:
        fields[k] = " ".join(fields[k])

    return fields


# 🔹 Fallback rule-based extraction (jab model blank de)
def fallback_rules(words):
    text = " ".join(words)

    data = {
        "VALUE": "",
        "START_DATE": "",
        "END_DATE": "",
        "RENEWAL": "",
        "PARTY1": "",
        "PARTY2": ""
    }

    # 💰 VALUE
    m = re.search(r'₹?\s?\d[\d,]*', text)
    if m:
        data["VALUE"] = m.group()

    # 📅 DATES
    dates = re.findall(r'\d{1,2}\s[A-Za-z]{3,9}\s\d{4}', text)
    if len(dates) >= 1:
        data["START_DATE"] = dates[0]
    if len(dates) >= 2:
        data["END_DATE"] = dates[1]

    # 🔁 RENEWAL
    m = re.search(r'\d+\s?days', text, re.IGNORECASE)
    if m:
        data["RENEWAL"] = m.group()

    return data


# 🔹 Predict single image
def predict_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    image, words, boxes = ocr_with_boxes(image_path)

    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**encoding)

    preds = outputs.logits.argmax(-1).squeeze().tolist()

    results = decode_predictions(words, preds[:len(words)])

    # 🔁 fallback if model gives empty output
    if all(v == "" for v in results.values()):
        results = fallback_rules(words)

    return results


# 🔹 Run on test folder
if __name__ == "__main__":

    test_folder = "test_images"

    if not os.path.exists(test_folder):
        raise FileNotFoundError("❌ test_images folder not found")

    for img in os.listdir(test_folder):

        if img.endswith(".png"):

            path = os.path.join(test_folder, img)

            try:
                preds = predict_image(path)

                print(f"\n📄 {img}")
                for k, v in preds.items():
                    print(f"{k}: {v}")

            except Exception as e:
                print(f"❌ Error in {img}:", e)
