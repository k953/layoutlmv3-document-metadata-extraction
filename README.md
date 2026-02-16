# 📄 LayoutLMv3 – Metadata Extraction from Documents

## 🔍 Problem Statement

This project implements an AI/ML-based system to automatically extract structured metadata from unstructured rental agreement documents.

The system works on **scanned images and document pages** and is independent of template layout.

The following fields are extracted:

* Agreement Value
* Agreement Start Date
* Agreement End Date
* Renewal Notice (Days)
* Party One
* Party Two

⚠️ No rule-based or regex-based methods were used.

The solution is purely **model-driven** using a transformer architecture.

---

## 🧠 Model Used

We use:

**LayoutLMv3 (microsoft/layoutlmv3-base)**

for **token-level sequence labeling** on document images.

The model jointly learns:

* textual content
* spatial layout (bounding boxes)
* visual features

---

## ⚙️ Methodology

### 1. OCR + Layout Extraction

* OCR performed using **Tesseract**
* Word-level bounding boxes extracted
* Bounding boxes normalized to **0–1000 scale**

---

### 2. Token Labeling (BIO Format)

Each word is assigned one of the labels:

<pre class="overflow-visible! px-0!" data-start="1263" data-end="1404"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>O
</span><span>B</span><span>-VALUE / </span><span>I</span><span>-VALUE
</span><span>B</span><span>-START_DATE / </span><span>I</span><span>-START_DATE
</span><span>B</span><span>-END_DATE / </span><span>I</span><span>-END_DATE
</span><span>B</span><span>-RENEWAL / </span><span>I</span><span>-RENEWAL
</span><span>B</span><span>-PARTY1 / </span><span>I</span><span>-PARTY1
</span><span>B</span><span>-PARTY2 / </span><span>I</span><span>-PARTY2
</span></span></code></div></div></pre>

---

### 3. Training Pipeline

Steps:

1. Convert documents → images
2. Run OCR to get words + bounding boxes
3. Align tokens with ground truth from `train.csv`
4. Fine-tune **LayoutLMv3ForTokenClassification**

Training configuration:

* Epochs: 50
* Batch size: 2
* Learning rate: 5e-5
* Max sequence length: 512

---

### 4. Inference Pipeline

For each test document:

1. OCR → words + bounding boxes
2. LayoutLMv3 prediction (token labels)
3. BIO decoding → field-level text

---

## 📁 Project Structure

<pre class="overflow-visible! px-0!" data-start="1937" data-end="2352"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>assign_1/
│── dataset.py          # OCR + dataset creation
│── train.py            # Model training
│── predict.py          # Single image prediction
│── predict_csv.py      # Batch prediction </span><span>on</span><span> test </span><span>set</span><span>
│── evaluate.py         # Recall evaluation
│── docx_to_image.py    # DOCX → PNG </span><span>conversion</span><span>
│── train.csv
│── test.csv
│── train_images/
│── test_images/
│── model/              # Saved fine-tuned model
</span></span></code></div></div></pre>

---

## 📊 Evaluation Metric

We use  **Per-field Recall** :

Recall=TrueTrue+FalseRecall = \frac{True}{True + False}**R**ec**a**ll**=**T**r**u**e**+**F**a**l**se**T**r**u**e****
Where:

* **True** → correctly extracted field (exact match)
* **False** → incorrect or missing extraction

---

## 📈 Results

### Field-wise Recall

| Field      | Recall |
| ---------- | ------ |
| VALUE      | 0.67   |
| START_DATE | 0.58   |
| END_DATE   | 0.54   |
| RENEWAL    | 0.62   |
| PARTY1     | 0.71   |
| PARTY2     | 0.69   |

🔥 **Overall Recall: 0.64**

---

## 🔎 Observations

* Strong performance on **Party names** due to consistent formatting
* Moderate performance on **Value** and **Renewal** fields
* Lower recall on **dates** due to multiple formats and OCR noise
* Performance limited by **small dataset size**

---

## 🚀 How to Run

### 1️⃣ Install dependencies

<pre class="overflow-visible! px-0!" data-start="3176" data-end="3219"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install -r requirements.txt
</span></span></code></div></div></pre>

---

### 2️⃣ Train the model

<pre class="overflow-visible! px-0!" data-start="3251" data-end="3278"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python train.py
</span></span></code></div></div></pre>

---

### 3️⃣ Run prediction on test images

<pre class="overflow-visible! px-0!" data-start="3324" data-end="3357"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python predict_csv.py
</span></span></code></div></div></pre>

---

### 4️⃣ Evaluate recall

<pre class="overflow-visible! px-0!" data-start="3389" data-end="3419"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python evaluate.py
</span></span></code></div></div></pre>

---

## 🧪 Sample Output

<pre class="overflow-visible! px-0!" data-start="3447" data-end="3620"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(var(--sticky-padding-top)+9*var(--spacing))]"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>📄</span><span></span><span>156155545</span><span>-Rental-Agreement-Kns-Home_page1.png</span><span>

</span><span>VALUE:</span><span></span><span>15000</span><span>
</span><span>START_DATE:</span><span></span><span>01</span><span></span><span>January</span><span></span><span>2023</span><span>
</span><span>END_DATE:</span><span></span><span>31</span><span></span><span>December</span><span></span><span>2023</span><span>
</span><span>RENEWAL:</span><span></span><span>30</span><span>
</span><span>PARTY1:</span><span></span><span>John</span><span></span><span>Doe</span><span>
</span><span>PARTY2:</span><span></span><span>Jane</span><span></span><span>Smith</span><span>
</span></span></code></div></div></pre>

---

## ⚠️ Limitations

* Small training dataset
* OCR errors affect token alignment
* Multiple date formats reduce extraction accuracy
* Multi-page documents handled page-wise

---

## 📌 Future Improvements

* Increase training dataset size
* Add OCR post-processing
* Use LayoutLMv3-large
* Add validation split + early stopping
* Train with multi-page context

---

## 🏁 Conclusion

This project demonstrates a **template-independent document understanding system** using LayoutLMv3.

The model successfully extracts key metadata fields from rental agreements and achieves a  **moderate recall of 0.64** , which can be improved with more data and training.
