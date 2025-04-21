# MMRPhys
Efficient and Robust Multidimensional Attention in Remote Physiological Sensing through Target Signal Constrained Factorization

Visit live demo of the web-app deploying MMRPhys: https://physiologicailab.github.io/mmrphys-live/

# About the Repository

## Relative paths of codes related to core contributions - MMRPhys and TSFM

        ├── neural_methods
        │   ├── model
        │   │   ├── MMRPhys
        │   │   │   ├── MMRPhysLEF.py
        │   │   │   ├── MMRPhysMEF.py
        │   │   │   ├── MMRPhysSEF.py
        │   │   │   ├── TSFM.py
        │   └── trainer
        │       ├── MMRPhysTrainer.py

## :notebook: Algorithms

The repo currently supports the following algorithms:

* Supervised Neural Algorithms
  * MMRPhys with TSFM - our proposed method
  * [FactorizePhys with FSAM](https://proceedings.neurips.cc/paper_files/paper/2024/file/af1c61e4dd59596f033d826419870602-Paper-Conference.pdf), by Joshi *et al.*, 2024
  * [Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks (PhysNet)](https://bmvc2019.org/wp-content/uploads/papers/0186-paper.pdf), by Yu *et al.*, 2019
  * [EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement (EfficientPhys)](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf), by Liu *et al.*, 2023
  * [EfficientPhys with FSAM (adapted for TSM network)](https://proceedings.neurips.cc/paper_files/paper/2024/file/af1c61e4dd59596f033d826419870602-Paper-Conference.pdf), by Joshi *et al.*, 2024
  * [PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer (PhysFormer)](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_PhysFormer_Facial_Video-Based_Physiological_Measurement_With_Temporal_Difference_Transformer_CVPR_2022_paper.pdf), by Yu *et al.*, 2022

## :file_folder: Datasets

The repo supports five datasets, namely iBVP, PURE, SCAMPS, UBFC-rPPG, and BP4D+. **To use these datasets in a deep learning model, you should organize the files as follows.**

* [iBVP](https://github.com/PhysiologicAILab/iBVP-Dataset)
  * Joshi, J.; Cho, Y. iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels. Electronics 2024, 13, 1334.
    -----------------
          iBVP_Dataset/
          |   |-- p01_a/
          |      |-- p01_a_rgb/
          |      |-- p01_a_t/
          |      |-- p01_a_bvp.csv
          |   |-- p01_b/
          |      |-- p01_b_rgb/
          |      |-- p01_b_t/
          |      |-- p01_b_bvp.csv
          |...
          |   |-- pii_x/
          |      |-- pii_x_rgb/
          |      |-- pii_x_t/
          |      |-- pii_x_bvp.csv
    -----------------

* [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)
  * Stricker, R., Müller, S., Gross, H.-M.Non-contact "Video-based Pulse Rate Measurement on a Mobile Service Robot" in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
    -----------------
         data/PURE/
         |   |-- 01-01/
         |      |-- 01-01/
         |      |-- 01-01.json
         |   |-- 01-02/
         |      |-- 01-02/
         |      |-- 01-02.json
         |...
         |   |-- ii-jj/
         |      |-- ii-jj/
         |      |-- ii-jj.json
    -----------------

* [SCAMPS](https://arxiv.org/abs/2206.04197)
  * D. McDuff, M. Wander, X. Liu, B. Hill, J. Hernandez, J. Lester, T. Baltrusaitis, "SCAMPS: Synthetics for Camera Measurement of Physiological Signals", NeurIPS, 2022
    -----------------
         data/SCAMPS/Train/
            |-- P00001.mat
            |-- P00002.mat
         |...
         data/SCAMPS/Val/
            |-- P00001.mat
            |-- P00002.mat
         |...
         data/SCAMPS/Test/
            |-- P00001.mat
            |-- P00002.mat
         |...
    -----------------

* [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)
  * S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
    -----------------
         data/UBFC-rPPG/
         |   |-- subject1/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |   |-- subject2/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |...
         |   |-- subjectn/
         |       |-- vid.avi
         |       |-- ground_truth.txt
    -----------------
 
* [BP4D+](https://sites.google.com/view/ybenezeth/ubfcrppg)
  * Zheng Zhang, Jeff Girard, Yue Wu, Xing Zhang, Peng Liu, Umur Ciftci, Shaun Canavan, Michael Reale, Andy Horowitz, Huiyuan Yang, Jeff Cohn, Qiang Ji, and Lijun Yin, "Multimodal Spontaneous Emotion Corpus for Human Behavior Analysis", CVPR 2016
    -----------------
         data/BP4D/
         |   |-- F001_T01/
         |      |-- F001_T01_rgb/
         |      |-- F001_T01_t/
         |      |-- F001_T01_phys.csv
         |   |-- F001_T02/
         |      |-- F001_T02_rgb/
         |      |-- F001_T02_t/
         |      |-- F001_T02_phys.csv
         |...
         |   |-- piii_xx/
         |      |-- piii_xx_rgb/
         |      |-- piii_xx_t/
         |      |-- piii_xx_phys.csv
    -----------------


## :wrench: Setup (First Time)

STEP 1: `chmod +x setup_env.sh`

STEP 2: `./setup_env.sh`

## :wrench: Activate Env

STEP 3: `conda activate mmrphys`

## :computer: Example of Using Pre-trained Models

Please use config files under `./configs/infer_configs`

For example, if you want to run The model trained on iBVP and tested on UBFC-rPPG, use `python main.py --config_file configs/infer_configs/BVP/Cross/RGB/iBVP_UBFC-rPPG_MMRPhys_SFSAM_Label.yaml`

## :computer: Examples of Neural Network Training

Please use config files under `./configs/train_configs`

### Training on iBVP and Testing on PURE With MMRPhys

STEP 1: Download the PURE raw data by asking the [paper authors](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure).

STEP 2: Download the iBVP raw data by asking the [paper authors](https://github.com/PhysiologicAILab/iBVP-Dataset).

STEP 3: Modify `configs/train_configs/BVP/Cross/RGB/iBVP_PURE_MMRPhys_SFSAM_Label.yaml`

STEP 4: Run `python main.py --config_file configs/train_configs/BVP/Cross/RGB/iBVP_PURE_MMRPhys_SFSAM_Label.yaml`

Note 1: Preprocessing requires only once; thus turn it off on the yaml file when you train the network after the first time.

Note 2: The example yaml setting will allow 100% of PURE to train and and test on iBVP after training 10 for epochs. Alternatively, this can be changed to train using 80% of PURE, validate with 20% of PURE and use the best model(with the least validation loss) to test on iBVP.
