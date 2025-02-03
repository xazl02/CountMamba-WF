# CountMamba

This repository is official implementation of:

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

**All formatted datasets can be downloaded at https://zenodo.org/records/14195051**

## 4. Website FingerPrinting
<details>
  
<summary>Code</summary>

### 4.1 DL-WF
```shell
cd DL-WF
for method in AWF DF RF TF TikTok TMWF VarCNN
do
  python main.py --dataset CW --train_epochs 100 --config config/${method}.ini
  python test.py --dataset CW --config config/${method}.ini --load_ratio 100 --result_file test_p100
  
  python main.py --dataset OW --train_epochs 100 --config config/${method}.ini
  python test.py --dataset OW --config config/${method}.ini --load_ratio 100 --result_file test_p100
  
  python main.py --dataset k-NN --train_epochs 100 --config config/${method}.ini
  python test.py --dataset k-NN --config config/${method}.ini --load_ratio 100 --result_file test_p100
  
  python main.py --dataset W_T --train_epochs 100 --config config/${method}.ini
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
for dataset in CW OW W_T
do
  python main.py --dataset ${dataset} --log_transform --maximum_load_time 120 --max_matrix_len 2700
  python test.py --dataset ${dataset} --log_transform --load_ratio 100 --result_file test_p100 --maximum_load_time 120 --max_matrix_len 2700
done

python main.py --dataset k-NN --log_transform --seq_len 10000 --maximum_load_time 120 --max_matrix_len 2700
python test.py --dataset k-NN --log_transform --seq_len 10000 --load_ratio 100 --result_file test_p100 --maximum_load_time 120 --max_matrix_len 2700

```

</details>

![image](https://github.com/SJTU-dxw/CountMamba-WF/blob/main/img/single_tab_result.png)

## 5. WF for defensed traffic
<details>
  
<summary>Code</summary>

### 5.1 DL-WF
```shell
cd DL-WF
for method in AWF DF RF TF TikTok TMWF VarCNN
do
  for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW
  do
    python main.py --dataset ${dataset} --train_epochs 100 --config config/${method}.ini
    python test.py --dataset ${dataset} --config config/${method}.ini --load_ratio 100 --result_file test_p100
  done
done
```

### 5.2 ML-WF
```shell
cd ML-WF
for dataset in wtfpad_CW front_CW tamaraw_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW
do
  python k-FP.py --dataset ${dataset}
  python k-FP_test.py --dataset ${dataset} --load_ratio 100 --result_file test_p100
  
  python CUMUL.py --dataset ${dataset}  # Set max_iter=100,0000 for trafficsilver_bwr_CW
  python CUMUL_test.py --dataset ${dataset} --load_ratio 100 --result_file test_p100
done
```

### 5.3 CountMamba
```shell
cd CountMamba
for dataset in wtfpad_CW front_CW regulator_CW trafficsilver_rb_CW trafficsilver_bd_CW trafficsilver_bwr_CW
do
  python main.py --dataset ${dataset} --log_transform --maximum_load_time 120 --max_matrix_len 2700
  python test.py --dataset ${dataset} --log_transform --load_ratio 100 --result_file test_p100 --maximum_load_time 120 --max_matrix_len 2700
done

python main.py --dataset tamaraw_CW --log_transform --seq_len 10000 --maximum_load_time 120 --max_matrix_len 2700
python test.py --dataset tamaraw_CW --log_transform --seq_len 10000 --load_ratio 100 --result_file test_p100 --maximum_load_time 120 --max_matrix_len 2700
```

</details>

![image](https://github.com/SJTU-dxw/CountMamba-WF/blob/main/img/defense_result.png)

## 6. WF for early-stage detection
<details>
  
<summary>Code</summary>

Generate early-stage test set
```shell
cd data_process
python gen_early_traffic.py --dataset CW
```

### 6.1 Holmes
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
python train.py --dataset CW --train_epochs 100
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
### 6.2 DL-WF
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
### 6.3 ML-WF
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

### 6.4 CountMamba
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

</details>

![image](https://github.com/SJTU-dxw/CountMamba-WF/blob/main/img/early_stage_result.png)

### 6.5 Real-World Early-Stage Traffic Classification
<details>
  
<summary>Code</summary>

```shell
cd EarlyStage
for threshold in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0
do
  python main.py --dataset CW --model CountMamba --threshold ${threshold} --device cuda --maximum_load_time 120 --max_matrix_len 2700 --embed_dim 256
done

for device in cuda
do
  for model in RF AWF DF TMWF TikTok VarCNN
  do
    for threshold in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0
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
</details>

![image](https://github.com/SJTU-dxw/CountMamba-WF/blob/main/img/realworld_earlystage_result.png)


## 7. WF for multi-tab detection
<details>
  
<summary>Code</summary>

### 7.1 DL-WF
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

### 7.2 CountMamba
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
</details>

![image](https://github.com/SJTU-dxw/CountMamba-WF/blob/main/img/multi_tab_result.png)


### 7.3 Fine-grained multi-tab detection
<details>
  
<summary>Code</summary>

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
</details>

![image](https://github.com/SJTU-dxw/CountMamba-WF/blob/main/img/finegrained_multitab_result.png)

## 8. Ablation Study on 2-tab Closed-world
### 8.1 Embedding Dimension
| Embedding Dim | 32    | 64    | 128   | 256   |
| ------------- | ----- | ----- | ----- | ----- |
| Closed        | 80.69 | 87.51 | 90.06 | 91.81 |
| Open          | 79.80 | 85.95 | 88.73 | 90.17 | 

### 8.2 Maximum Loading Time
| Maximum Loading Time | 160   | 240   | 320   | 480   | 640   |
| -------------------- | ----- | ----- | ----- | ----- | ----- |
| Closed               | 91.19 | 91.67 | 91.81 | 91.55 |       |
| Open                 | 89.65 | 90.43 | 90.17 | 89.82 |       | 

### 8.3 Time Window Length
|  Time Window Length  |  11   |  22   |  44   |  88   |  132  |
| -------------------- | ----- | ----- | ----- | ----- | ----- |
| Closed               |       |       | 91.81 |       |       |
| Open                 |       |       | 90.17 |       |       |

