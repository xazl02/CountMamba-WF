# CountMamba-WF

## 1. Denpendency
``` shell
conda create -n myenv python=3.10
conda activate myenv
# pytorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# Mamba-ssm
wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# causal-conv1d
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# other packages
pip install tqdm
pip install pytorch_metric_learning
pip install captum
pip install pandas
pip install timm
pip install natsort
pip install noise
```

## 2. Dataset
### 2.1 Download Dataset
Prepare following datasets in "dataset" folder:
- DF: provided by Tik-Tok
  - CW (https://zenodo.org/records/11631265/files/Undefended.zip?download=1)
  - OW (https://zenodo.org/records/11631265/files/Undefended_OW.zip?download=1)
- Walkie-Talkie:
  - https://zenodo.org/records/11631265/files/W_T_Simulated.zip?download=1
- k-NN: 
  - https://github.com/kdsec/wangknn-dataset
  
- Multi-Tab datasets:
  - ARES Dataset
    - https://github.com/Xinhao-Deng/Multitab-WF-Datasets
  - TMWF Dataset
    - https://github.com/jzx-bupt/TMWF
### 2.2 Process Raw Dataset (k-NN & W_T & OW)
```shell
cd data_process
python concat_cell.py  # k-NN & W_T
python check_format.py  # Manually modify the tail of the illegal file OW/5278340744671043543057
```

### 2.3 Merge Traces (TMWF)
```shell
cd data_process
python MergeSingleTraces_openworld.py --input_dir "/nvme/dxw/TMWF-main/dataset/tbb single tab/" --output_dir "/nvme/dxw/TMWF-main/dataset/tbb_multi_tab/"
python MergeSingleTraces_openworld.py --input_dir "/nvme/dxw/TMWF-main/dataset/chrome single tab/" --output_dir "/nvme/dxw/TMWF-main/dataset/chrome_multi_tab/"
```

### 2.4 Convert to npz
data.npz
```shell
cd data_process
python convert_to_npz.py --dataset CW
python convert_to_npz.py --dataset OW
python convert_to_npz.py --dataset k-NN
python convert_to_npz.py --dataset W_T

python convert_multi_tab_npz.py --dataset Closed_2tab
python convert_multi_tab_npz.py --dataset Closed_3tab
python convert_multi_tab_npz.py --dataset Closed_4tab
python convert_multi_tab_npz.py --dataset Closed_5tab
python convert_multi_tab_npz.py --dataset Open_2tab
python convert_multi_tab_npz.py --dataset Open_3tab
python convert_multi_tab_npz.py --dataset Open_4tab
python convert_multi_tab_npz.py --dataset Open_5tab

python convert_merge_npz.py --input_file "/nvme/dxw/TMWF-main/dataset/tbb_multi_tab/"
python convert_merge_npz.py --input_file "/nvme/dxw/TMWF-main/dataset/chrome_multi_tab/"
```

### Todo
Provide npz dataset

### 2.5 Dataset Split
train.npz, valid.npz, test.npz
```shell
cd data_process
python dataset_split.py --dataset CW
python dataset_split.py --dataset OW
python dataset_split.py --dataset k-NN
python dataset_split.py --dataset W_T

python dataset_split.py --dataset Closed_2tab --use_stratify False
python dataset_split.py --dataset Closed_3tab --use_stratify False
python dataset_split.py --dataset Closed_4tab --use_stratify False
python dataset_split.py --dataset Closed_5tab --use_stratify False
python dataset_split.py --dataset Open_2tab --use_stratify False
python dataset_split.py --dataset Open_3tab --use_stratify False
python dataset_split.py --dataset Open_4tab --use_stratify False
python dataset_split.py --dataset Open_5tab --use_stratify False

python dataset_split.py --dataset tbb_multi_tab --use_stratify False
python dataset_split.py --dataset chrome_multi_tab --use_stratify False
```

## 3. Early-Stage WF

生成Early-Traffic测试集
```shell
cd data_process
python gen_early_traffic.py --dataset CW
python gen_early_traffic.py --dataset OW
python gen_early_traffic.py --dataset k-NN
python gen_early_traffic.py --dataset W_T
```

### 3.1 Holmes
temporal_train.py, temporal_valid.py
```shell
cd Holmes
python temporal_extractor.py --dataset CW --in_file train
python temporal_extractor.py --dataset CW --in_file valid
```
RF_IS/max_f1.pth
```shell
cd Holmes
python train_RF.py --dataset CW --train_epochs 30
```
attr_DeepLiftShap.npz
```shell
cd Holmes
python feature_attr.py --dataset CW
```
aug_train.npz, aug_valid.npz
```shell
cd Holmes
python data_augmentation.py --dataset CW --in_file train
python data_augmentation.py --dataset CW --in_file valid
```
taf_aug_train.npz, taf_aug_valid.npz
```shell
cd Holmes
python gen_taf.py --dataset CW --in_file aug_train
python gen_taf.py --dataset CW --in_file aug_valid
```
Holmes/max_f1.pth
```shell
cd Holmes
python train.py --dataset CW --train_epochs 30
```
spatial_distribution.npz
```shell
cd Holmes
python spatial_analysis.py --dataset CW
```
taf_test_p10.npz, ..., taf_test_p100.npz
```shell
cd Holmes
python gen_taf.py --dataset CW --in_file test_p10
python gen_taf.py --dataset CW --in_file test_p20
python gen_taf.py --dataset CW --in_file test_p30
python gen_taf.py --dataset CW --in_file test_p40
python gen_taf.py --dataset CW --in_file test_p50
python gen_taf.py --dataset CW --in_file test_p60
python gen_taf.py --dataset CW --in_file test_p70
python gen_taf.py --dataset CW --in_file test_p80
python gen_taf.py --dataset CW --in_file test_p90
python gen_taf.py --dataset CW --in_file test_p100
```
test_p10.json, ..., test_p100.json
```shell
cd Holmes
python test.py --dataset CW --test_file taf_test_p10 --result_file test_p10
python test.py --dataset CW --test_file taf_test_p20 --result_file test_p20
python test.py --dataset CW --test_file taf_test_p30 --result_file test_p30
python test.py --dataset CW --test_file taf_test_p40 --result_file test_p40
python test.py --dataset CW --test_file taf_test_p50 --result_file test_p50
python test.py --dataset CW --test_file taf_test_p60 --result_file test_p60
python test.py --dataset CW --test_file taf_test_p70 --result_file test_p70
python test.py --dataset CW --test_file taf_test_p80 --result_file test_p80
python test.py --dataset CW --test_file taf_test_p90 --result_file test_p90
python test.py --dataset CW --test_file taf_test_p100 --result_file test_p100
```
### 3.2 DL-WF
```shell
cd DL-WF
python main.py --dataset CW --train_epochs 30 --config config/AWF.ini
python main.py --dataset CW --train_epochs 30 --config config/DF.ini
python main.py --dataset CW --train_epochs 30 --config config/RF.ini
python main.py --dataset CW --train_epochs 30 --config config/TF.ini
python main.py --dataset CW --train_epochs 30 --config config/TikTok.ini
python main.py --dataset CW --train_epochs 30 --config config/TMWF.ini
python main.py --dataset CW --train_epochs 30 --config config/VarCNN.ini
```

```shell
cd DL-WF
for percent in {10..100..10}
do
  python test.py --dataset CW --config config/AWF.ini --load_ratio ${percent} --result_file test_p${percent}
  python test.py --dataset CW --config config/DF.ini --load_ratio ${percent} --result_file test_p${percent}
  python test.py --dataset CW --config config/RF.ini --load_ratio ${percent} --result_file test_p${percent}
  python test.py --dataset CW --config config/TF.ini --load_ratio ${percent} --result_file test_p${percent}
  python test.py --dataset CW --config config/TikTok.ini --load_ratio ${percent} --result_file test_p${percent}
  python test.py --dataset CW --config config/TMWF.ini --load_ratio ${percent} --result_file test_p${percent}
  python test.py --dataset CW --config config/VarCNN.ini --load_ratio ${percent} --result_file test_p${percent}
done
```
### 3.3 ML-WF
```shell
cd ML-WF
python k-FP.py --dataset CW
python CUMUL.py --dataset CW
```

```shell
cd ML-WF
for percent in {10..100..10}
do
  python k-FP_test.py --dataset CW --load_ratio ${percent} --result_file test_p${percent}
  python CUMUL_test.py --dataset CW --load_ratio ${percent} --result_file test_p${percent}
 done 
```

### 3.4 CountMamba
```shell
cd CountMamba
python main.py --dataset CW --log_transform --num_aug 50
```

```shell
cd CountMamba
for percent in {10..100..10}
do
  python test.py --dataset CW --log_transform --load_ratio ${percent} --result_file test_p${percent}
done
```
### 3.5 Results on Early-Stage Traffic
| Loaded     | 10%   | 20%   | 30%   | 40%   | 50%   | 60%   | 70%   | 80%   | 90%   | 100%  |
| ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| k-FP       | 0.43  | 2.71  | 5.35  | 8.37  | 14.59 | 26.13 | 42.59 | 63.71 | 81.36 | 88.66 |
| CUMUL      | 2.42  | 7.28  | 12.56 | 20.19 | 31.71 | 50.57 | 71.06 | 85.94 | 94.17 | 97.37 |
| AWF        | 5.07  | 12.91 | 22.82 | 36.64 | 51.97 | 66.05 | 79.48 | 87.65 | 93.13 | 95.45 |
| TF         | 5.15  | 12.73 | 23.41 | 39.18 | 56.14 | 70.9  | 83.1  | 91.25 | 96.48 | 97.96 |
| TMWF       | 3.59  | 9.0   | 15.95 | 27.09 | 42.95 | 61.81 | 77.53 | 89.89 | 95.89 | 97.28 |
| DF         | 6.37  | 15.12 | 26.23 | 42.83 | 58.95 | 73.45 | 86.05 | 94.0  | 97.46 | 98.47 |
| TikTok     | 4.96  | 13.07 | 22.53 | 36.31 | 53.75 | 69.89 | 84.02 | 92.81 | 97.22 | 98.53 |
| VarCNN     | 7.11  | 16.29 | 30.23 | 48.15 | 64.15 | 77.54 | 86.84 | 93.74 | 97.73 | 98.74 |
| RF         | 11.22 | 27.5  | 49.05 | 71.34 | 84.54 | 91.62 | 95.28 | 97.33 | 98.47 | 98.67 |
| Holmes     | 23.87 | 52.97 | 76.40 | 90.00 | 94.67 | 96.27 | 96.95 | 97.74 | 98.31 | 98.48 |
| CountMamba | 31.95 | 58.82 | 79.67 | 90.18 | 94.49 | 96.35 | 97.38 | 97.85 | 98.3  | 98.65 |

### 3.6 Real-World Early-Stage Traffic Classification
```shell
cd EarlyStage
for device in cuda cpu
do
  for model in CountMamba RF AWF DF TMWF TikTok VarCNN
  do
    for threshold in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.01
    do
      python main.py --dataset CW --model ${model} --threshold ${threshold} --device ${device}
    done
  done
done
```

```shell
cd EarlyStage
for device in cuda cpu
do
  for threshold in -0.2 -0.15 -0.1 -0.07 -0.05 -0.02 0.0 0.02 0.05 0.1
  do
    python main.py --dataset CW --model Holmes --threshold ${threshold} --device ${device}
  done
done
```

## 4. WF on Other Datasets

### 4.1 DL-WF
```shell
cd DL-WF
for method in AWF DF RF TF TikTok TMWF VarCNN
do
  python main.py --dataset CW --train_epochs 30 --config config/${method}.ini
  python test.py --dataset CW --config config/${method}.ini --load_ratio 100 --result_file test_p100
  
  python main.py --dataset OW --train_epochs 30 --config config/${method}.ini
  python test.py --dataset OW --config config/${method}.ini --load_ratio 100 --result_file test_p100
  
  python main.py --dataset k-NN --train_epochs 100 --config config/${method}.ini
  python test.py --dataset k-NN --config config/${method}.ini --load_ratio 100 --result_file test_p100
  
  python main.py --dataset W_T --train_epochs 30 --config config/${method}.ini
  python test.py --dataset W_T --config config/${method}.ini --load_ratio 100 --result_file test_p100
done
```

### 4.2 ML-WF
```shell
cd ML-WF
for dataset in CW OW k-NN W_T
do
  python k-FP.py --dataset ${dataset}
  python k-FP_test.py --dataset ${dataset} --load_ratio 100 --result_file test_p100
  
  python CUMUL.py --dataset ${dataset}
  python CUMUL_test.py --dataset ${dataset} --load_ratio 100 --result_file test_p100
done
```

### 4.3 CountMamba
```shell
cd CountMamba
for dataset in CW OW k-NN W_T
do
  python main.py --dataset ${dataset} --log_transform --entire_sequence
  python test.py --dataset ${dataset} --log_transform --load_ratio 100 --result_file test_p100
done
```

## 5. Multi-Tab WF


## 6. Defence
- WTF-PAD: Add dummy packets. No latency.
  ``` shell
  cd defense/wtfpad
  python main.py --traces_path "../../dataset/CW"
  ```
- FRONT: Add dummy packets with fixed length of 888. No latency.
  ``` shell
  cd defense/front
  python main.py --p "../../dataset/CW"
  ```
- Tamaraw: Send packets at constant rate with fixed size.
  ``` shell
  cd defense/tamaraw
  python tamaraw.py --traces_path "../../dataset/CW"
  ```
- RegularTor: transmit packets in a time-sensitive manner
  - When a download traffic 'surge' arrives, RegulaTor starts sending packets at a set initial rate.
  - If no packets are available when one is scheduled, a dummy packet is sent instead.
  - At the same time, RegulaTor sends upload packets at some fraction of the download packet sending rate.
  ``` shell
  cd defense/regulartor
  python regulator_sim.py --source_path "../../dataset/CW/" --output_path "../results/regulator_CW/"
  ```
- TrafficSilver: Split traffic.
  - Round Robin
    ``` shell
    cd defense/trafficsilver
    python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_rb_CW/" --s round_robin
    ```
  - By Direction
    ``` shell
    cd defense/trafficsilver
    python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_bd_CW/" --s in_and_out
    ```
  - Batched Weighted Random
    ``` shell
    cd defense/trafficsilver
    python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_bwr_CW/" --s batched_weighted_random -r 50,70 -a 1,1,1
    ```

### 6.1 Convert to npz
``` shell
cd data_process
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW
do
  python convert_to_npz.py --dataset ${dataset}
done
```

### 6.2 Dataset Split
```shell
cd data_process
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW
do
  python dataset_split.py --dataset ${dataset}
done
```

### 6.3 Early Traffic
```shell
cd data_process
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW
do
  python gen_early_traffic.py --dataset ${dataset}
done
```

### 6.4 DL-WF
```shell
cd DL-WF
for method in AWF DF RF TF TikTok TMWF VarCNN
do
  for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW
  do
    python main.py --dataset ${dataset} --train_epochs 30 --config config/${method}.ini
    python test.py --dataset ${dataset} --config config/${method}.ini --load_ratio 100 --result_file test_p100
  done
done
```

### 6.5 ML-WF
```shell
cd ML-WF
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW
do
  python k-FP.py --dataset ${dataset}
  python k-FP_test.py --dataset ${dataset} --load_ratio 100 --result_file test_p100
  
  python CUMUL.py --dataset ${dataset}
  python CUMUL_test.py --dataset ${dataset} --load_ratio 100 --result_file test_p100
done
```

### 6.6 CountMamba
```shell
cd CountMamba
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW
do
  python main.py --dataset ${dataset} --log_transform --entire_sequence
  python test.py --dataset ${dataset} --log_transform --load_ratio 100 --result_file test_p100
done
```

## 7. Closed-World Evaluation on defensed datasets

### 7.1 CountMamba-WF

### 7.2 DL-based Methods

### 7.3 ML-based Methods

## 8. Countermeasure
- To be done
