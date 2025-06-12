# SAE-FiRE: Sparse Autoencoder for Financial Representation Enhancement

<embed src="./diagram/architecture7.pdf" type="application/pdf" width="100%" height="600px" />

## 1. Data Acquisition
- Raw transcript JSONL files and metadata CSVs  
- Financial datasets (GVKEY mappings, surprises, Compustat, IBES)

## 2. Preprocessing
- Transcript metadata construction 
- S&P 500 and financial link data expansion 

## 3. Feature Extraction
- SAE feature encoding via pretrained Gemma Scope models  
- Last hidden state feature extraction from last hidden states from Gemma 2 for use in baseline models

## 4. Feature Selection & Classification
- ANOVA and tree-based ranking methods  
- Training LR on top features

## 5. Experiments & Results
- **SAE Variants:** Gemma-2-2b (16 k features), Gemma-2-9b (131 k features)  
- **Feature Selection (SAE):** Top-k features where  
  - $k \in \{500, 1000, 1500, 2000, 2500\}$ for SAE on Gemma-2-2b  
  - $k \in \{3000, 3500, 4000, 4500, 5000, 5500, 6000\}$ for SAE on Gemma-2-9b  
- **Baseline Models:**  
  1. SAE features **without** selection  
     - MLP 
     - XGBoost  
  2. Gemma-2 last hidden state
     - MLP  
     - LR 