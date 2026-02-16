from predict import predict_image
import pandas as pd
import os

FIELDS = [
    "VALUE",
    "START_DATE",
    "END_DATE",
    "RENEWAL",
    "PARTY1",
    "PARTY2"
]

df = pd.read_csv("test.csv")

true_count = {f: 0 for f in FIELDS}
pred_count = {f: 0 for f in FIELDS}


def normalize(text):
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


for _, row in df.iterrows():

    file_name = os.path.splitext(str(row["File Name"]).strip())[0]

    folder = os.path.join("test_images", file_name)

    if not os.path.exists(folder):
        print("⚠ Missing folder:", folder)
        continue

    gt = {
        "VALUE": normalize(row.get("Agreement Value")),
        "START_DATE": normalize(row.get("Agreement Start Date")),
        "END_DATE": normalize(row.get("Agreement End Date")),
        "RENEWAL": normalize(row.get("Renewal Notice (Days)")),
        "PARTY1": normalize(row.get("Party One")),
        "PARTY2": normalize(row.get("Party Two")),
    }

    for img in os.listdir(folder):

        if not img.endswith(".png"):
            continue

        image_path = os.path.join(folder, img)

        try:
            preds = predict_image(image_path)
        except Exception as e:
            print("❌ Error:", image_path, e)
            continue

        for field in FIELDS:

            gt_val = gt[field]
            pred_val = normalize(preds.get(field, ""))

            if gt_val == "":
                continue

            true_count[field] += 1

            if gt_val in pred_val:
                pred_count[field] += 1


print("\n📊 FIELD-WISE RECALL:\n")

overall_true = 0
overall_pred = 0

for field in FIELDS:

    t = true_count[field]
    p = pred_count[field]

    overall_true += t
    overall_pred += p

    recall = p / t if t > 0 else 0

    print(f"{field:12} → Recall: {recall:.3f}  ({p}/{t})")

overall_recall = overall_pred / overall_true if overall_true > 0 else 0

print("\n🔥 OVERALL RECALL:", round(overall_recall, 3))
