#  Final AI Project

This repository contains the student-facing materials for the AI final project. It is designed to provide a consistent starting point while leaving the core implementation to the student.

---

##  Quick Start
To get started, please review the following documents:

* **Project Overview:** [`starter_pack/README.md`](starter_pack/README.md)
* **Starter Checklist:** [`starter_pack/CHECKLIST.md`](starter_pack/CHECKLIST.md)

---

##  Repository Contents
The following structure defines the organization of this project:

| Path | Description |
| :--- | :--- |
| `deliverables/ai_project.tex` | The official handout and project requirements. |
| `starter_pack/data/` | Fixed datasets and pre-defined split indices. |
| `starter_pack/scripts/` | Deterministic data-preparation and synthetic-data utilities. |
| `starter_pack/report/template.tex` | An optional LaTeX template for your final report. |
| `starter_pack/src/` | **Main Workspace:** Minimal scaffold for your model code. |
| `starter_pack/figures/` | Directory for generated plots and visualizations. |
| `starter_pack/results/` | Directory for saving model outputs and performance metrics. |
| `starter_pack/slides/` | Minimal scaffold for your presentation slides. |

---

##  Starter-Pack Policy
The starter pack is **intentionally minimal**. It focuses on data consistency and organization rather than providing logic.

###  What is Included:
* Fixed digits features, labels, and split indices.
* Deterministic split and synthetic-data scripts.
* A minimal repository skeleton and README/Checklist.
* An optional report template.

###  What is NOT Included:
* Model architecture code (Neural Networks, Softmax, etc.).
* Training-loop scaffolding or evaluation logic.
* Optimizer implementations.
* Hidden solutions or pre-trained weights.

---

##  Suggested First Steps
Follow these steps to ensure a smooth project start:

1.  **Documentation:** Review the starter-pack README and checklist thoroughly.
2.  **Setup:** Clone the repository and create your dedicated **working branches**.
3.  **Data Inspection:** Use the provided scripts to inspect the `.npz` files before you begin writing model code.
4.  **Environment:** Ensure all dependencies (like `numpy` and `scikit-learn`) are installed.

---
