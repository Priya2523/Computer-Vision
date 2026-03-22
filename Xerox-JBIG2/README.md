Xerox Assignment Files 
# 🖨️ Xerox JBIG2 Compression Bug Simulation

## 📌 What is the Bug?
In 2013, it was discovered that Xerox photocopiers using JBIG2 compression
were silently changing numbers in scanned documents.

Example:
- The number **6** was being replaced by **8**
- The number **9** was being replaced by **3**

This happened because JBIG2 reuses similar-looking symbols to save storage space.
The scary part — the image looked completely identical to human eye!

---

## 🎯 Objective
Simulate the JBIG2 compression bug in Python and prove its impact using:
- Visual comparison (before vs after)
- Image quality metrics (SSIM, PSNR)
- OCR accuracy comparison
- Financial impact analysis
- Entropy-based fix

---

## 📁 Files

| File | Description |
|---|---|
| `xerox_compression_bug_CV.ipynb` | Main simulation notebook |

---

## 📓 Notebook Structure

| Cell | Description |
|---|---|
| Cell 1 | Introduction & what we will do |
| Cell 2 | Install & import libraries |
| Cell 3-4 | Create original document with invoice numbers |
| Cell 5-6 | Simulate JBIG2 bug (digit substitution) |
| Cell 7-8 | Side by side visual comparison |
| Cell 9-10 | SSIM & PSNR quality metrics |
| Cell 11-12 | OCR accuracy before vs after |
| Cell 13-14 | Financial impact analysis |
| Cell 15-16 | Safe compression rule (the fix) |
| Cell 17-18 | Final summary & conclusion |

---

## 📊 Simulation Results

| Metric | Value |
|---|---|
| SSIM Score | ~0.85+ (images look identical!) |
| OCR Lines Wrong | 3 out of 3 (100% error rate!) |
| Financial Error | ₹5,99,798 difference! |
| Detectability | Nearly impossible without source |

---

## ⚠️ Real World Impact

- **Medical records** → wrong dosage amounts
- **Legal documents** → wrong contract values
- **Bank statements** → wrong transaction amounts
- **Tax documents** → wrong filing numbers

---

## 💡 How to Fix

1. Use **lossless compression** for critical documents
2. Set **strict entropy threshold** before substitution
3. Always verify scanned documents with original
4. Use **digital signatures** to detect tampering

---

## 🛠️ Libraries Used

- OpenCV
- NumPy
- Matplotlib
- scikit-image (SSIM)
- pytesseract (OCR)

-

**Priya A**
AI & Data Science 
Computer Vision Assignment
