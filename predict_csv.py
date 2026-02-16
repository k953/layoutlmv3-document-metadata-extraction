import torch
from transformers import LayoutLMv3ForTokenClassification
from dataset import ocr_with_boxes, processor, id2label

model = LayoutLMv3ForTokenClassification.from_pretrained("./model")
model.eval()

def extract_fields(image_path):
    image, words, boxes = ocr_with_boxes(image_path)

    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length"
    )

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()

    result = {
        "VALUE": "",
        "START_DATE": "",
        "END_DATE": "",
        "RENEWAL": "",
        "PARTY1": "",
        "PARTY2": ""
    }

    for word, pred in zip(words, predictions[:len(words)]):
        label = id2label[pred]
        if label.startswith("B-"):
            key = label.replace("B-", "")
            result[key] += word + " "

    return {k: v.strip() for k, v in result.items()}
