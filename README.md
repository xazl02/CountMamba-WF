# CountMamba

This reposity is official implementation of:

***CountMamba: A Generalized Website Fingerprinting Attack via Coarse-Grained Representation and Fine-Grained Prediction***

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
<details>
  
<summary>Prepare Dataset</summary>

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

### 2.5 Dataset Split
```shell
cd data_process
for dataset in CW OW k-NN W_T
do
  python dataset_split.py --dataset ${dataset}
done

for dataset in Closed_2tab Closed_3tab Closed_4tab Closed_5tab Open_2tab Open_3tab Open_4tab Open_5tab tbb_multi_tab chrome_multi_tab
do
  python dataset_split.py --dataset ${dataset} --use_stratify False
done
```

</details>

## 3. Defense Dataset
<details>
  
<summary>Prepare Dataset</summary>

### 3.1 Defenses
- WTF-PAD: Add dummy packets. No latency.
  ``` shell
  cd defense/wtfpad
  python main.py --traces_path "../../dataset/CW"
  python main.py --traces_path "../../dataset/OW"

  cd defense_npz/wtfpad
  python main.py --traces_path "../../npz_dataset/Closed_2tab"
  python main.py --traces_path "../../npz_dataset/Open_2tab"
  ```
- FRONT: Add dummy packets with fixed length of 888. No latency.
  ``` shell
  cd defense/front
  python main.py --p "../../dataset/CW"
  python main.py --p "../../dataset/OW"

  cd defense_npz/front
  python main.py --p "../../npz_dataset/Closed_2tab"
  python main.py --p "../../npz_dataset/Open_2tab"
  ```
- Tamaraw: Send packets at constant rate with fixed size.
  ``` shell
  cd defense/tamaraw
  python tamaraw.py --traces_path "../../dataset/CW"
  ```
- RegulaTor: transmit packets in a time-sensitive manner
  - When a download traffic 'surge' arrives, RegulaTor starts sending packets at a set initial rate.
  - If no packets are available when one is scheduled, a dummy packet is sent instead.
  - At the same time, RegulaTor sends upload packets at some fraction of the download packet sending rate.
  ``` shell
  cd defense/regulartor
  python regulator_sim.py --source_path "../../dataset/CW/" --output_path "../results/regulator_CW/"
  python regulator_sim.py --source_path "../../dataset/OW/" --output_path "../results/regulator_OW/"

  cd defense_npz/regulartor
  python regulator_sim.py --source_path "../../npz_dataset/Closed_2tab" --output_path "../results/regulator_Closed_2tab"
  python regulator_sim.py --source_path "../../npz_dataset/Open_2tab" --output_path "../results/regulator_Open_2tab"
  ```
- TrafficSilver: Split traffic.
  - Round Robin
    ``` shell
    cd defense/trafficsilver
    python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_rb_CW/" --s round_robin
    python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_rb_OW/" --s round_robin
    ```
  - By Direction
    ``` shell
    cd defense/trafficsilver
    python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_bd_CW/" --s in_and_out
    python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_bd_OW/" --s in_and_out
    ```
  - Batched Weighted Random
    ``` shell
    cd defense/trafficsilver
    python simulator.py --p "../../dataset/CW/" --o "../results/trafficsilver_bwr_CW/" --s batched_weighted_random -r 50,70 -a 1,1,1
    python simulator.py --p "../../dataset/OW/" --o "../results/trafficsilver_bwr_OW/" --s batched_weighted_random -r 50,70 -a 1,1,1
    ```

### 3.2 Overhead for defense methods (CW)

| Defense           | Latency Overhead | Bandwith Overhead |
| ----------------- | ---------------- | ----------------- |
| WTF-PAD           | 1.00             | 1.47              |
| FRONT             | 1.00             | 1.46              |
| Tamaraw           | 2.82             | 3.69              |
| RegulaTor         | 1.05             | 1.58              |
| TrafficSilver-RB  | 1.00             | 1.00              |
| TrafficSilver-BD  | 1.00             | 1.00              |
| TrafficSilver-BWR | 1.00             | 1.00              | 

### 3.3 Convert to npz
``` shell
cd data_process
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW wtfpad_OW front_OW regulator_OW trafficsilver_rb_OW trafficsilver_bd_OW trafficsilver_bwr_OW
do
  python convert_to_npz.py --dataset ${dataset}
done
```

### 3.4 Dataset Split
```shell
cd data_process
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW wtfpad_OW front_OW regulator_OW trafficsilver_rb_OW trafficsilver_bd_OW trafficsilver_bwr_OW
do
  python dataset_split.py --dataset ${dataset}
done
```

```shell
for dataset in wtfpad_Closed_2tab wtfpad_Open_2tab front_Closed_2tab front_Open_2tab regulator_Closed_2tab regulator_Open_2tab
do
  python dataset_split.py --dataset ${dataset} --use_stratify False
done
```

</details>

## 3. Website FingerPrinting

### 3.1 DL-WF
```shell
cd DL-WF
for method in AWF DF RF TF TikTok TMWF VarCNN
do
  python main.py --dataset CW --train_epochs 30 --config config/${method}.ini
  python test.py --dataset CW --config config/${method}.ini --load_ratio 100 --result_file test_p100
  
  python main.py --dataset OW --train_epochs 30 --config config/${method}.ini
  python test.py --dataset OW --config config/${method}.ini --load_ratio 100 --result_file test_p100
  
  python main.py --dataset k-NN --train_epochs 50 --config config/${method}.ini
  python test.py --dataset k-NN --config config/${method}.ini --load_ratio 100 --result_file test_p100
  
  python main.py --dataset W_T --train_epochs 30 --config config/${method}.ini
  python test.py --dataset W_T --config config/${method}.ini --load_ratio 100 --result_file test_p100
done
```

### 3.2 ML-WF
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

### 3.3 CountMamba
```shell
cd CountMamba
for dataset in CW OW
do
  python main.py --dataset ${dataset} --log_transform --maximum_load_time 120 --max_matrix_len 2700
  python test.py --dataset ${dataset} --log_transform --load_ratio 100 --result_file test_p100 --maximum_load_time 120 --max_matrix_len 2700
done

for dataset in k-NN W_T
do
  python main.py --dataset ${dataset} --log_transform --maximum_load_time 120 --max_matrix_len 2700 --seq_len 10000
  python test.py --dataset ${dataset} --log_transform --load_ratio 100 --result_file test_p100 --maximum_load_time 120 --max_matrix_len 2700 --seq_len 10000
done

```

### 3.4 Results on Normal Traffic

| Dataset    | CW        | OW        | k-NN      | W_T       |
| ---------- | --------- | --------- | --------- | --------- |
| k-FP       | 88.64     | 84.35     | 58.95     | 76.26     |
| CUMUL      | 97.37     | 95.64     | 90.33     | 15.07     |
| AWF        | 95.45     | 94.27     | 75.30     | 25.41     |
| TF         | 97.96     | 94.63     | 53.19     | 47.29     |
| TMWF       | 97.28     | 96.22     | 79.25     | 37.03     |
| DF         | 98.47     | 97.71     | 87.12     | 37.36     |
| TikTok     | 98.53     | 97.66     | 84.64     | 96.90     |
| VarCNN     | 98.74     | 98.37     | 89.57     | **99.38** |
| RF         | 98.67     | 98.58     | 93.04     | **99.38** | 
| CountMamba | **98.97** | **98.68** | **94.72** | 99.23     |


## 4. WF for defensed traffic

### 4.2 DL-WF
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

### 4.3 ML-WF
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

### 4.4 CountMamba
```shell
cd CountMamba
for dataset in wtfpad_CW front_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW
do
  python main.py --dataset ${dataset} --log_transform  --maximum_load_time 120 --max_matrix_len 2700
  python test.py --dataset ${dataset} --log_transform --load_ratio 100 --result_file test_p100 --maximum_load_time 120 --max_matrix_len 2700
done

python main.py --dataset tamaraw_CW --log_transform --seq_len 10000  --maximum_load_time 120 --max_matrix_len 2700
python test.py --dataset tamaraw_CW --log_transform --seq_len 10000 --load_ratio 100 --result_file test_p100 --maximum_load_time 120 --max_matrix_len 2700
```

### 4.5 Results on Defensed Traffic

| Dataset    | WTF-PAD | Front | Tamaraw | RegulaTor | TrafficSilver-RB | TrafficSilver-BD | TrafficSilver-BWR | 
| ---------- | ------- | ----- | ------- | --------- | ---------------- | ---------------- | ----------------- |
| k-FP       |         |       |         |           |                  |                  |                   |
| CUMUL      |         |       |         |           |                  |                  |                   |
| AWF        |         |       |         |           |                  |                  |                   |
| TF         |         |       |         |           |                  |                  |                   |
| TMWF       |         |       |         |           |                  |                  |                   |
| DF         |         |       |         |           |                  |                  |                   |
| Tik-Tok    |         |       |         |           |                  |                  |                   |
| VarCNN     |         |       |         |           |                  |                  |                   |
| RF         |         |       |         |           |                  |                  |                   |
| CountMamba |         |       |         |           |                  |                  |                   |

## 5. WF for early-stage detection

Generate early-stage test set
```shell
cd data_process
python gen_early_traffic.py --dataset CW
```

### 5.1 Holmes
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
### 5.2 DL-WF
**Skip this step if you have already trained models in 3.1 DL-WF**
```shell
cd DL-WF 
python main.py --dataset CW --train_epochs 100 --config config/AWF.ini
python main.py --dataset CW --train_epochs 100 --config config/DF.ini
python main.py --dataset CW --train_epochs 100 --config config/RF.ini
python main.py --dataset CW --train_epochs 100 --config config/TF.ini
python main.py --dataset CW --train_epochs 100 --config config/TikTok.ini
python main.py --dataset CW --train_epochs 100 --config config/TMWF.ini
python main.py --dataset CW --train_epochs 100 --config config/VarCNN.ini
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
### 5.3 ML-WF
**Skip this step if you have already trained models in 3.2 ML-WF**
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

### 5.4 CountMamba
```shell
cd CountMamba
python main.py --dataset CW --log_transform --early_stage --num_aug 50 --maximum_load_time 120 --max_matrix_len 2700
```

```shell
cd CountMamba
for percent in {10..100..10}
do
  python test.py --dataset CW --log_transform --load_ratio ${percent} --result_file test_p${percent} --maximum_load_time 120 --max_matrix_len 2700
done
```
### 5.5 Results on Early-Stage Traffic (F1-score)
| Loaded     | 10%   | 20%   | 30%   | 40%   | 50%   | 60%   | 70%   | 80%   | 90%   | 100%  |
| ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| k-FP       | 0.43  | 2.71  | 5.35  | 8.37  | 14.59 | 26.13 | 42.59 | 63.71 | 81.36 | 88.64 |
| CUMUL      | 2.42  | 7.28  | 12.56 | 20.19 | 31.71 | 50.57 | 71.06 | 85.94 | 94.17 | 97.37 |
| AWF        | 4.44  | 10.74 | 19.5  | 32.3  | 46.99 | 63.01 | 77.65 | 87.11 | 93.11 | 95.83 |
| TF         | 6.23  | 15.24 | 28.76 | 46.32 | 62.14 | 75.22 | 85.69 | 92.77 | 97.02 | 98.46 |
| TMWF       | 4.19  | 11.08 | 19.6  | 31.34 | 44.56 | 62.32 | 79.57 | 90.82 | 96.19 | 97.63 |
| DF         | 7.49  | 16.35 | 28.19 | 45.26 | 61.97 | 75.78 | 87.07 | 94.0  | 97.58 | 98.6  |
| Tik-Tok    | 6.72  | 14.65 | 24.44 | 39.57 | 56.98 | 72.35 | 85.2  | 93.05 | 97.29 | 98.7  |
| Var-CNN    | 7.58  | 17.3  | 31.97 | 50.1  | 66.0  | 78.78 | 88.57 | 94.27 | 97.76 | 98.76 |
| RF         | 12.83 | 28.87 | 50.01 | 70.86 | 84.67 | 91.92 | 95.5  | 97.62 | 98.68 | 99.0  |
| Holmes     | 24.23 | 53.3  | 77.39 | 89.94 | 94.81 | 96.27 | 96.91 | 97.83 | 98.34 | 98.59 |
| CountMamba | 32.81 | 58.68 | 79.75 | 90.73 | 94.9  | 96.72 | 97.51 | 98.07 | 98.53 | 98.85 |

### 5.6 Real-World Early-Stage Traffic Classification
```shell
cd EarlyStage
for threshold in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  python main.py --dataset CW --model CountMamba --threshold ${threshold} --device cuda --maximum_load_time 120 --max_matrix_len 2700 --embed_dim 256
done

for device in cuda
do
  for model in RF AWF DF TMWF TikTok VarCNN
  do
    for threshold in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
      python main.py --dataset CW --model ${model} --threshold ${threshold} --device ${device}
    done
  done
done
```

```shell
cd EarlyStage
for device in cuda
do
  for threshold in 0.0
  do
    python main.py --dataset CW --model Holmes --threshold ${threshold} --device ${device}
  done
done
```

| Threshold  | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.01 |
| ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
| AWF        |     |     |     |     |     |     |     |     |     |     |     |      |
| TMWF       |     |     |     |     |     |     |     |     |     |     |     |      |
| DF         |     |     |     |     |     |     |     |     |     |     |     |      |
| TikTok     |     |     |     |     |     |     |     |     |     |     |     |      |
| VarCNN     |     |     |     |     |     |     |     |     |     |     |     |      |
| CountMamba |     |     |     |     |     |     |     |     |     |     |     |      |


## 6. WF for multi-tab detection

### 6.1 DL-WF
```shell
cd DL-WF
for method in ARES TMWF AWF DF MultiTab_RF TikTok VarCNN
do
  for num in 2 3 4 5
  do
    python main.py --dataset Closed_${num}tab --train_epochs 100 --config config/${method}.ini --num_tabs ${num}
    python test.py --dataset Closed_${num}tab --config config/${method}.ini --load_ratio 100 --result_file test_p100 --num_tabs ${num}
    
    python main.py --dataset Open_${num}tab --train_epochs 100 --config config/${method}.ini --num_tabs ${num}
    python test.py --dataset Open_${num}tab --config config/${method}.ini --load_ratio 100 --result_file test_p100 --num_tabs ${num}
  done
done
```

### 6.2 CountMamba
```shell
cd CountMamba
for num in 2 3 4 5
do
  python main.py --dataset Closed_${num}tab --log_transform --num_tabs ${num} --seq_len 10000 --maximum_load_time 320 --max_matrix_len 7200
  python test.py --dataset Closed_${num}tab --log_transform --num_tabs ${num} --seq_len 10000 --maximum_load_time 320 --max_matrix_len 7200 --load_ratio 100 --result_file test_p100
  
  python main.py --dataset Open_${num}tab --log_transform --num_tabs ${num} --seq_len 10000 --maximum_load_time 320 --max_matrix_len 7200
  python test.py --dataset Open_${num}tab --log_transform --num_tabs ${num} --seq_len 10000 --maximum_load_time 320 --max_matrix_len 7200 --load_ratio 100 --result_file test_p100
done
```

### 6.3 Results on Multi-Tab Traffic
| Dataset     | Closed_2tab | Closed_3tab | Closed_4tab | Closed_5tab | Open_2tab | Open_3tab | Open_4tab | Open_5tab |
| ----------- | ----------- | ----------- | ----------- | ----------- | --------- | --------- | --------- | --------- |
| AWF         |             |             |             |             |           |           |           |           |
| TMWF        |             |             |             |             |           |           |           |           |
| DF          |             |             |             |             |           |           |           |           |
| TikTok      |             |             |             |             |           |           |           |           |
| VarCNN      |             |             |             |             |           |           |           |           |
| MultiTab_RF |             |             |             |             |           |           |           |           |
| ARES        |             |             |             |             |           |           |           |           |
| CountMamba  |             |             |             |             |           |           |           |           |


### 6.4 Fine-grained multi-tab detection
```shell
cd DL-WF
for method in ARES TMWF AWF DF MultiTab_RF TikTok VarCNN
do
  for num in 2
  do
    python main.py --dataset tbb_multi_tab --train_epochs 300 --config config/${method}.ini --num_tabs ${num}
    python test.py --dataset tbb_multi_tab --config config/${method}.ini --load_ratio 100 --result_file test_p100 --num_tabs ${num}
    
    python main.py --dataset chrome_multi_tab --train_epochs 300 --config config/${method}.ini --num_tabs ${num}
    python test.py --dataset chrome_multi_tab --config config/${method}.ini --load_ratio 100 --result_file test_p100 --num_tabs ${num}
  done
done
```

```shell
cd CountMamba
for num in 2
do
  python main.py --dataset tbb_multi_tab --epochs 300 --log_transform --num_tabs ${num} --seq_len 10000 --maximum_load_time 160 --max_matrix_len 3600 --fine_predict
  python test.py --dataset tbb_multi_tab --log_transform --num_tabs ${num} --seq_len 10000 --maximum_load_time 160 --max_matrix_len 3600 --load_ratio 100 --result_file test_p100 --fine_predict
  
  python main.py --dataset chrome_multi_tab --epochs 300 --log_transform --num_tabs ${num} --seq_len 10000 --maximum_load_time 160 --max_matrix_len 3600 --fine_predict
  python test.py --dataset chrome_multi_tab --log_transform --num_tabs ${num} --seq_len 10000 --maximum_load_time 160 --max_matrix_len 3600 --load_ratio 100 --result_file test_p100 --fine_predict
done
```
