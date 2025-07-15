# LAGCN: Local Aggregation Graph Convolutional Network for Video-based Human Action Recognition

## Abstract
Human Action Recognition (HAR) is a critical task in computer vision with applications in surveillance, virtual reality (VR), augmented reality (AR), and human-computer interaction (HCI). In this work, we propose a novel Local Aggregation Graph Convolutional Network (LAGCN) for video-based human action recognition. The model combines local attention, global attention, and temporal modeling to effectively capture spatial and temporal features of human actions using graph convolutional networks (GCN). Experiments on the Multiview 3D and UCF-101 datasets demonstrate that LAGCN outperforms existing methods in terms of accuracy, F1 score, and mAP, while also providing superior computational efficiency in terms of inference speed and lower GFLOPs.

## Overview
This repository contains the code and experimental results for our proposed LAGCN model. The model is designed to improve the performance of human action recognition tasks using video data, leveraging the power of graph convolutional networks combined with attention mechanisms and temporal modeling.

### Key Contributions:
- **LAGCN Model**: Combines local attention, global attention, and temporal modeling to improve recognition accuracy.
- **Efficient Computation**: Achieves higher accuracy and F1 score while maintaining low GFLOPs and a high inference speed.
- **Extensive Evaluation**: Demonstrates the effectiveness of LAGCN on two widely used human action recognition datasets: **Multiview 3D** and **UCF-101**.

## Code
The code provided in this repository implements the LAGCN model for human action recognition. Key components of the model include:
1. **2D Pose Estimation Module**: Using HRNet to estimate 2D pose coordinates.
2. **Local Attention Module**: Captures local dependencies between joints using GCN and local attention mechanisms.
3. **Global Attention Module**: Models global human pose using GAT for self-adaptive feature aggregation.
4. **Temporal Modeling Module**: Uses Dilated Temporal Convolutions (DTC) for capturing temporal dependencies.

You can execute the code to train and evaluate the model on the Multiview 3D and UCF-101 datasets.

## Datasets
- **Multiview 3D Dataset**: Captured at UCLA with synchronized Kinect cameras, this dataset contains 10 different action categories and provides RGB images, depth maps, and skeleton data.
  - [Download the Multiview 3D Dataset](https://wangjiangb.github.io/my_data.html)

- **UCF-101 Dataset**: A widely used action recognition dataset with 13,320 videos spanning 101 action categories. It is available for benchmarking on YouTube video-based actions.
  - [Download the UCF-101 Dataset](https://www.crcv.ucf.edu/research/data-sets/ucf101/)

## Experiment Results
The model was evaluated on two widely used datasets:
1. **Multiview 3D Dataset**: LAGCN achieved an accuracy of 91.2%, outperforming other methods.
2. **UCF-101 Dataset**: LAGCN achieved an accuracy of 92.7% and F1 score of 92.6%, demonstrating strong generalization on real-world video data.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LAGCN.git
   cd LAGCN
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the dataset paths and ensure the datasets are downloaded as described above.

## Usage

To train the model on your dataset, use the following command:

```bash
python train.py --dataset Multiview3D --batch_size 16 --epochs 50 --learning_rate 0.0001
```

To test the model:

```bash
python test.py --model LAGCN --dataset UCF101
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

* The **Multiview 3D Dataset** and **UCF-101 Dataset** are provided for the purpose of benchmarking and evaluating human action recognition methods.
* Special thanks to the authors of the datasets and other related works that contributed to the research in this field.

## Contact

For any questions, feel free to reach out at `sunmaojin0427@163.com`.


### Key Points Covered:
- Title and Abstract of your work.
- Overview of the LAGCN model and its contributions.
- Code and instructions on how to set up and use the repository.
- Dataset links to Multiview 3D and UCF-101, along with download links.
- Installation instructions, including Python package dependencies.
- Instructions for training and testing the model.
- License and Contact details.
