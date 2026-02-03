# BakeNet: 지각적 품질 최적화 초해상도 모델 (Perceptually Optimized Image Super-Resolution)

BakeNet은 Oklab 색공간을 활용하여 손실된 색 정보를 복원하고 이미지를 업스케일링하는 AI 모델입니다. 단순한 픽셀 매칭보다 사람이 실제로 느끼는 색감의 정확도(지각적 품질)에 초점을 맞추었습니다.

## 🚀 빠른 시작 (클라우드 환경)

### 1. 환경 설정
저장소를 복제하고 설정 스크립트를 실행하여 필요한 라이브러리를 설치합니다.

```bash
git clone https://github.com/Tom-Heo/killthetopaz.git
cd killthetopaz
bash setup.sh
bash download_data.sh
python train.py --data_root ./data
```

### 2. 데이터셋 준비 (DIV2K)
DIV2K 데이터셋을 자동으로 다운로드하고 압축을 해제합니다.

```bash
bash download_data.sh
```

---

## 🛠 사용법

### 학습 (Training)

처음부터 학습을 시작합니다:
```bash
git pull
python train.py --data_root ./data
```

**주요 옵션:**
*   `--batch_size`: 배치 크기 (기본값: 16). VRAM 부족 시 줄이세요.
*   `--epochs`: 총 에폭 수 (기본값: 100).
*   `--save_interval`: 체크포인트 저장 간격 (기본값: 10 에폭).

**학습 재개 (Resume):**
저장된 체크포인트에서 학습을 이어가려면:
```bash
git pull
python train.py --data_root ./data --resume checkpoints/bake_last.pth
```

### 추론 (Inference / Upscaling)

학습된 모델을 사용하여 이미지를 업스케일링합니다. 결과물은 기본적으로 `results/` 폴더에 저장됩니다.

**단일 이미지:**
```bash
python inference.py --input "test.png" --checkpoint "checkpoints/bake_best.pth" --pre_upsample
```

**폴더 내 모든 이미지:**
```bash
python inference.py --input "images_folder/" --checkpoint "checkpoints/bake_best.pth" --pre_upsample --output_dir "my_results/"
```

*   `--pre_upsample`: 입력 이미지가 작은 저해상도 이미지(예: 썸네일)인 경우 사용하세요. 모델 입력 전 Bicubic으로 4배 확대합니다.
*   `--output_dir`: 결과 이미지가 저장될 경로를 지정합니다. (기본값: `results/`)
*   입력 이미지가 이미 확대된 상태(흐릿하지만 해상도는 큼)라면 `--pre_upsample` 옵션을 빼세요.

---

## 📂 프로젝트 구조

*   `heo.py`: 핵심 모델 정의 (BakeNet, BakeDDPM, HeLU, BakedColor).
*   `train.py`: 학습 스크립트 (EMA, AdamW, Step LR 적용).
*   `inference.py`: 업스케일링 추론 스크립트.
*   `dataset.py`: DIV2K 데이터 로더 (Oklab 자동 변환).
*   `setup.sh`: 리눅스 환경 자동 설정 스크립트.
*   `download_data.sh`: DIV2K 데이터셋 다운로더.
