# One-Shot Handwritten Character Classifier with BPL

A Python implementation of a one-shot handwritten character classifier using **Bayesian Program Learning (BPL)** and **modified Hausdorff distance** as a similarity metric. This script evaluates the ability to classify new characters based on only one example, simulating a "one-shot" learning scenario.

Inspired by the original research work from [Brenden Lake's GitHub repository](https://github.com/brendenlake/BPL).

<img src="https://github.com/pau-lo/Bayesian-Program-Learning-BPL/blob/master/assets/BPL.png" width="1200" alt="human and machine learning-BPL">

[Art by Danqing Wang]

> Here is a brief explanation of what is [BPL algorithm](http://www.ctvnews.ca/sci-tech/new-algorithm-lets-machines-learn-like-humans-1.2695230).

----

## Table of Contents

- [One-Shot Handwritten Character Classifier with BPL](#-one-shot-handwritten-character-classifier-with-bpl)
  - [Table of Contents](#-table-of-contents)
  - [Overview](#-overview)
  - [Features](#-features)
  - [Approach \& Methodology](#-approach--methodology)
    - [Key Steps:](#key-steps)
  - [Requirements](#-requirements)
  - [Directory Structure](#-directory-structure)
  - [How to Use](#-how-to-use)
    - [Step-by-step Instructions:](#step-by-step-instructions)
  - [Example Output](#-example-output)
  - [Citation \& Acknowledgments](#-citation--acknowledgments)
    - [Research Paper:](#research-paper)
    - [Original GitHub Code:](#original-github-code)
  - [License](#-license)

---

## Overview

This project implements a one-shot handwritten character classification system, drawing inspiration from the **Bayesian Program Learning (BPL)** framework. The classifier leverages pixel coordinates and modified Hausdorff distance to compare shapes and determine similarity between characters.

The implementation is based on an open-source MATLAB version by [Brenden Lake](https://github.com/brendenlake/BPL), which has been translated into Python for easier use and integration with modern data science workflows.

---

## Features

- **One-shot learning**: Classify new characters using only a single example.
- **Modified Hausdorff distance**: A robust measure of shape similarity between two sets of coordinates.
- **Multiple runs**: Average error rate is computed across multiple independent classification experiments.
- **Modular and extensible design** for easy customization.

---

## Approach & Methodology

The classifier follows the principles of **Bayesian Program Learning (BPL)** as described in [Lake et al. (2015)](#citation), where characters are represented by their pixel coordinates, and classification is performed through similarity comparisons using a cost function.

### Key Steps:
1. Load image pixel coordinates.
2. Compute modified Hausdorff distance between test and training samples.
3. Select the most similar training example as the predicted class.
4. Evaluate performance using percentage error rate across multiple runs.

---

## Requirements

Make sure to install the following dependencies:

```bash
pip install numpy scipy imageio
```

---

## Directory Structure

The project expects a specific directory structure for input data:

```
one-shot-classifier/
│
├── data/all_runs/               # Contains training and test data
│   ├── run01/
│   ├── run02/
│   └── ...                 # Each run has its own label file (`class_labels.txt`)
│
├── README.md               # This file
└── main.py  # Main implementation script
```

> ⚠️ **Note:** All image files and label files should be placed in the `all_runs/` directory.

---

## How to Use

### Step-by-step Instructions:

1. **Prepare your data**:
   - Store images in the `all_runs/` directory with corresponding subfolders.
   - Ensure each folder has a file named `class_labels.txt`, containing pairs of test and training image paths, separated by spaces.

2. **Run the classifier**:

```bash
python main.py
```

3. **Output**:
   - The script will run multiple classification tests (default: 20 runs), displaying error rates for each.
   - It concludes with an average error rate across all runs.

---

## Example Output

Here’s what you might see after running the classifier:

```bash
Running one-shot handwritten character classifier
Run 01: Error rate 45.0%
Run 02: Error rate 35.0%
...
[RESULT] Average error rate across 20 runs: 38.8%
```

---

## Citation & Acknowledgments

This project is inspired by the original work of [Brenden Lake](https://github.com/brendenlake/BPL) and is adapted into Python for modern use.

### Research Paper:

> Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015).  
> **Human-level concept learning through probabilistic program induction**.  
> *Science*, 350(6266), 1332-1338.  
> [DOI: 10.1126/science.aab3027](https://doi.org/10.1126/science.aab3027)

### Original GitHub Code:

> This implementation is based on the MATLAB code from [Brenden Lake's BPL repository](https://github.com/brendenlake/BPL).

---

## License

This project is licensed under the **MIT License**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> This license allows for free use, modification, and distribution of this code, as long as you include the original copyright and license notice in any redistribution or derivative work.
