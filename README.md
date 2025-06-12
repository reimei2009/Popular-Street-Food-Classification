# 🍜 Popular Street Food Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](https://www.kaggle.com/code/reimei2009/street-food/notebook?scriptVersionId=245022592)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project that classifies images of popular street foods using a pre-trained **ResNet-50** architecture with **PyTorch**. The model achieves high accuracy through transfer learning, data augmentation, and hyperparameter optimization.

## 🎯 Project Highlights

- **High Performance**: Achieves excellent accuracy and F1 scores on street food classification
- **Transfer Learning**: Leverages pre-trained ResNet-50 for efficient training
- **Data Augmentation**: Robust training with image transformations
- **Comprehensive Analysis**: Includes EDA, training metrics, and visualizations
- **Production Ready**: Easy inference with confidence thresholds

## 📊 Demo & Results

🔗 **Live Demo**: [Street Food Classification Notebook on Kaggle](https://www.kaggle.com/code/reimei2009/street-food/notebook?scriptVersionId=245022592)

## 🗂️ Dataset

The dataset is sourced from Kaggle: [Popular Street Foods Dataset](https://www.kaggle.com/datasets/reimei2009/popular-street-foods)

**Dataset Structure:**
- Images organized by food categories (tacos, ramen, falafel, etc.)
- CSV file with dataset statistics
- Balanced distribution across classes
- High-quality street food images

## 🏗️ Project Architecture

```
street_food_classifier/
├── 📁 src/                     # Source code
│   ├── 📄 config.py            # Configuration & constants
│   ├── 📁 data/                # Data handling
│   │   ├── dataset.py          # Dataset loading & DataLoaders
│   │   ├── transforms.py       # Image transformations
│   │   └── eda.py              # Exploratory data analysis
│   ├── 📁 models/              # Model architecture
│   │   ├── model.py            # ResNet-50 initialization
│   │   └── train.py            # Training & evaluation
│   ├── 📁 utils/               # Utility functions
│   │   ├── file_utils.py       # File management
│   │   ├── metrics.py          # Metrics & visualization
│   │   └── visualization.py    # Prediction visualization
│   └── 📄 main.py              # Main pipeline
├── 📁 models/                  # Saved checkpoints
├── 📁 sample/                  # Visualizations & plots
├── 📄 requirements.txt         # Dependencies
└── 📄 README.md               # Documentation
```

### 📋 Component Details

| Component | Description |
|-----------|-------------|
| `config.py` | Global constants, hyperparameters, and reproducibility settings |
| `data/dataset.py` | Dataset loading, train/validation splitting, DataLoader creation |
| `data/transforms.py` | Image preprocessing and augmentation strategies |
| `data/eda.py` | Exploratory data analysis and distribution visualizations |
| `models/model.py` | ResNet-50 architecture adaptation for food classification |
| `models/train.py` | Training loop, evaluation, early stopping, checkpointing |
| `utils/file_utils.py` | Directory management and checkpoint utilities |
| `utils/metrics.py` | Performance metrics calculation and history plotting |
| `utils/visualization.py` | Prediction visualization and confusion matrix |
| `main.py` | Complete pipeline orchestration |

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended) or CPU
- **8GB+ RAM**
- **Sufficient storage** for dataset and models

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/reimei2009/street_food_classifier.git
   cd street_food_classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup dataset**
   - Download from [Kaggle Dataset](https://www.kaggle.com/datasets/reimei2009/popular-street-foods)
   - Place in `./data/popular_street_foods/dataset`
   - Ensure `dataset_stats.csv` is included

## 🎮 Usage

### Training the Model

Start training with default parameters:

```bash
python src/main.py
```

**Customization Options:**
- Modify hyperparameters in `src/config.py`
- Adjust data augmentation in `data/transforms.py`
- Monitor training progress through generated plots

### Making Predictions

```python
from src.utils.visualization import predict_image

# Predict single image
result = predict_image(
    image_path="path/to/your/image.jpg",
    model_path="models/best_f1_model.pth",
    threshold=0.4
)
print(f"Prediction: {result}")
```

### Hyperparameter Tuning

Key parameters to experiment with:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 0.001 | Controls training speed |
| `BATCH_SIZE` | 32 | Samples per batch |
| `EPOCHS` | 50 | Training iterations |
| `IMAGE_SIZE` | 224 | Input image dimensions |

## 🧠 Model Architecture

### ResNet-50 Overview

ResNet-50 is a 50-layer deep convolutional neural network featuring:

- **Residual Connections**: Skip connections prevent vanishing gradients
- **Bottleneck Blocks**: Efficient computation with 1x1, 3x3, 1x1 convolutions
- **Transfer Learning**: Pre-trained on ImageNet, fine-tuned for food classification
- **Feature Extraction**: Robust hierarchical feature learning

### Training Strategy

1. **Data Preprocessing**: Normalization and augmentation
2. **Transfer Learning**: Fine-tune pre-trained weights
3. **Optimization**: Adam optimizer with learning rate scheduling
4. **Regularization**: Early stopping and model checkpointing
5. **Evaluation**: Multi-metric assessment (accuracy, F1, loss)

## 📈 Performance Monitoring

The project tracks multiple metrics:

- **Training/Validation Loss**: Model convergence
- **Accuracy**: Classification performance
- **F1 Score**: Balanced precision and recall
- **Confusion Matrix**: Per-class performance analysis

All metrics are visualized and saved automatically during training.

## ☁️ Training on Kaggle

For users without local GPU resources:

1. **Access the [Kaggle Notebook](https://www.kaggle.com/code/reimei2009/street-food/notebook?scriptVersionId=245022592)**
2. **Fork the notebook** to your account
3. **Enable GPU acceleration** in notebook settings
4. **Run all cells** to start training
5. **Download trained models** for local inference

## 🔧 Advanced Configuration

### Custom Data Augmentation

```python
# In data/transforms.py
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    # Add your custom transforms here
])
```

### Model Checkpointing

The system automatically saves:
- **Best Loss Model**: Lowest validation loss
- **Best Accuracy Model**: Highest validation accuracy  
- **Best F1 Model**: Highest F1 score
- **Latest Checkpoint**: Most recent epoch

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or suggest features
- Submit pull requests
- Improve documentation
- Add new food categories

## 👨‍💻 Author

**Ngô Thanh Tình (reimei2009)**
- 🐙 GitHub: [@reimei2009](https://github.com/reimei2009)
- 📧 Email: thanhin875@gmail.com
- 🏆 Kaggle: [reimei2009](https://www.kaggle.com/reimei2009)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Created and provided by reimei2009 on Kaggle
- **Framework**: Built with PyTorch and torchvision
- **Inspiration**: Modern computer vision and transfer learning techniques
- **Community**: Kaggle community for feedback and support

## 📚 References

- [ResNet Paper](https://arxiv.org/abs/1512.03385) - Deep Residual Learning for Image Recognition
- [PyTorch Documentation](https://pytorch.org/docs/) - Official PyTorch documentation
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) - PyTorch transfer learning tutorial

---

<div align="center">

**🍕 Happy Street Food Classification! 🌮**

*If you found this project helpful, please consider giving it a ⭐*

</div>
