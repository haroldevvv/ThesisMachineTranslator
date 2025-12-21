# ThesisMachineTranslator

## AI-Driven Synthetic Data for Low-Resource Machine Translation  
### Enhancing Rinconada-to-English Translation

### Undergraduate Thesis Project

This repository contains the source code and experimental setup for the undergraduate thesis entitled:

> **AI-Driven Synthetic Data for Low-Resource Translation: Enhancing Rinconada-to-English Translation**

conducted by:

- **Harold M. Salvador**  
- **Mary Rachel L. Parañal**  
- **Paolo L. Espion**

and submitted to the **College of Computer Studies, Camarines Sur Polytechnic Colleges**,  
in partial fulfillment of the requirements for the degree of  
**Bachelor of Science in Computer Science**,  
**December 2025**.

---

## Project Overview

Low-resource languages present significant challenges in machine translation due to the scarcity of parallel corpora.  
This study investigates the use of **AI-driven synthetic data generation techniques** to improve translation quality for **Rinconada-to-English** machine translation.

The research explores and compares multiple translation strategies, including:

- Direct Machine Translation  
- Pivot-Based Machine Translation  
- Synthetic Data Augmentation  
- Prototype Translation Models  

The project emphasizes modular implementation, controlled experimentation, and reproducibility.

---

## Repository Structure

```text
ThesisMachineTranslator/
├── directMT/                  # Direct Rinconada-to-English MT implementation
├── direct_augmented_MT/        # Direct MT with synthetic data augmentation
├── pivotMT/                   # Pivot-based MT approach
├── pivot_augmented_MT/         # Pivot MT with synthetic data augmentation
├── Prototype_MT/              # Prototype and exploratory models
├── Datasets/                  # (Excluded from repository)
└── .gitignore
```text

> **Note:** Large files such as datasets, trained model weights, checkpoints, and experiment outputs are intentionally excluded from this repository.

---

## Datasets

Due to size limitations and data usage considerations, datasets are **not publicly included** in this repository.

📩 **Datasets are available upon reasonable request** for academic and research purposes.

---

## Reproducibility

All scripts for data preprocessing, training, and evaluation are provided.  
To reproduce the experiments, users must supply the appropriate datasets and execute the corresponding training pipelines as defined in each module.

---

## Acknowledgment

This research was conducted as part of an undergraduate thesis at **Camarines Sur Polytechnic Colleges**, under the guidance and supervision of the faculty of the **College of Computer Studies**.

---

## License

This project is intended for **academic and research use only**.  
Commercial use requires explicit permission from the authors.
