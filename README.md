# Efficient Lottery Ticket Hypothesis Pruning with Robust Training Techniques

## 📌 **Overview**
This project demonstrates the implementation of the **Lottery Ticket Hypothesis (LTH)** for neural network pruning using PyTorch. The model is trained on the MNIST dataset, and the impact of structured pruning is analyzed through iterative pruning and random reinitialization.

### 🚀 **Enhancements Implemented**
- **Gradient Clipping:** Prevents exploding gradients, ensuring stable training.
- **Dropout Regularization:** Reduces overfitting and enhances generalization.
- **Kaiming Initialization:** Facilitates better weight distribution for improved convergence.
- **AdamW Optimizer:** Enhances model performance with effective weight decay.
- **Reproducibility:** Fixed random seeds to ensure consistent experiment results.
- **Seaborn Visualizations:** Enhanced interpretability of accuracy and pruning effectiveness.

---

## 📦 **Installation**
Ensure you have Python and PyTorch installed. Then, run the following commands:

```bash
!pip install torch torchvision numpy matplotlib seaborn tqdm
```

Download the MNIST dataset automatically during runtime.

```bash
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
```

---

## 🛠️ **Usage**
1. Clone the repository:
```bash
git clone https://github.com/Harshin1V/Lottery-Ticket-Hypothesis.git
cd Lottery-Ticket-Hypothesis
```
2. Run the training and pruning process:
```bash
python Lottery-Ticket-Hypothesis.py
```

---

## 📊 **Results and Analysis**
- **Accuracy Comparison:**
  - Iterative Pruning consistently outperforms Random Reinitialization across different pruning levels.
- **Visualization:**
  - Using Seaborn, accuracy plots demonstrate the trade-off between sparsity and model performance.
  
---

## 📚 **Project Structure**
```
.
├── data
├── enhanced_lth_pruning.py
├── README.md
└── results
    ├── accuracy_plot_iterative.png
    ├── accuracy_plot_reinit.png
    └── comparison_plot.png
```
---

## 📝 **Acknowledgments**
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

---

## 📧 **Contact**
For any questions or feedback, reach out here: [Harshini](https://www.linkedin.com/in/harshini1v/)

