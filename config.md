# Conda Environment
  ## videorope-tf
    ```bash
      conda create -n videorope-tf python=3.10 -y
      conda activate videorope-tf

      pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

      pip install -U pip setuptools wheel packaging ninja

      pip install transformers==4.45.2 accelerate==0.34.2 tokenizers==0.20.1 safetensors==0.4.5 sentencepiece==0.2.0 numpy==1.26.1 pillow==10.4.0 requests==2.32.3 tqdm==4.67.1 av==13.1.0 decord==0.6.0 opencv-python==4.10.0.84 qwen-vl-utils==0.0.14

      MAX_JOBS=4 pip install flash-attn --no-build-isolation -v
    ```

  ## videorope-vllm
    ```bash
      conda create -n videorope-vllm python=3.10 -y
      conda activate videorope-vllm

      pip install -U pip setuptools wheel packaging ninja

      pip install 
        "huggingface_hub>=0.34,<1.0" 
        transformers==4.45.2 
        accelerate==0.34.2 
        tokenizers==0.20.1 
        safetensors==0.4.5 
        sentencepiece==0.2.0 
        numpy==1.26.1 
        pillow==10.4.0 
        requests==2.32.3 
        tqdm==4.67.1 
        av==13.1.0 
        decord==0.6.0 
        opencv-python==4.10.0.84 
        qwen-vl-utils==0.0.14

      pip install vllm

      pip install torchvision torchaudio

      MAX_JOBS=4 pip install flash-attn --no-build-isolation -v
    ```

# Checkpoint and Dataset
  ```bash
    python -m pip install -U "huggingface_hub[cli]"
    sudo apt-get update
    sudo apt-get install -y pv p7zip-full

    mkdir -p videorope_exp/{checkpoints,datasets}
    cd videorope_exp/
    
    # mirror
    export HF_ENDPOINT="https://hf-mirror.com"
  ```

  ## Checkpoint
    ```bash
      hf download Wiselnn/Qwen2-VL-videorope-128frames-8k-context-330k-llava-video --local-dir ./checkpoints/Qwen2-VL-videorope-128frames-8k-context-330k-llava-video
      hf download  Wiselnn/Qwen2-VL-m_rope-128frames-8k-context-330k-llava-video --local-dir ./checkpoints/Qwen2-VL-m_rope-128frames-8k-context-330k-llava-video
      hf download  Wiselnn/Qwen2-VL-tad_rope-128frames-8k-context-330k-llava-video --local-dir ./checkpoints/Qwen2-VL-tad_rope-128frames-8k-context-330k-llava-video
      hf download  Wiselnn/Qwen2-VL-vanilla_rope-128frames-8k-context-330k-llava-video --local-dir ./checkpoints/Qwen2-VL-vanilla_rope-128frames-8k-context-330k-llava-video
    ```

  ## Dataset
    ```bash
      hf auth login

      # LLaVA-Video-178k <3m
      hf download   --repo-type dataset   lmms-lab/LLaVA-Video-178K   --local-dir ./datasets/LLaVA-Video-178K_lt3m   --include "0_30_s_*/*" "30_60_s_*/*" "1_2_m_*/*" "2_3_m_*/*"

      # LongVideoBench
      hf download longvideobench/LongVideoBench --repo-type dataset --local-dir ./datasets/LongVideoBench
      cd $ROOT/datasets/LongVideoBench

      cat videos.tar.part.* | pv | tar --skip-old-files -xvf -
      pv subtitles.tar | tar --skip-old-files -xvf -

      # Video-MME
      hf download lmms-lab/Video-MME --repo-type dataset --local-dir $ROOT/datasets/Video-MME
      mkdir -p videos subtitles
      for z in videos_chunked_*.zip; do
        echo "==> extracting $z"
        7z x "$z" -o./videos -aos
      done
      7z x subtitle.zip -o./subtitles -aos
    ```

# VideoWeave-style WebVid-10K Pipeline

  ## download environment
    ```bash
      conda create -n download python=3.10 -y
      conda activate download
      python -m pip install --upgrade pip setuptools wheel
      pip install requests tqdm numpy opencv-python-headless
      python -m pip install -U "huggingface_hub[cli]"
    ```

  ## download WebVid metadata with URL
    ```bash
      cd $ROOT
      mkdir -p videorope_exp/datasets/webvid/metadata_hf_check

      hf download TempoFunk/webvid-10M \
        --repo-type dataset \
        --local-dir videorope_exp/datasets/webvid/metadata_hf_check \
        data/train/partitions/0322.csv

      head -n 5 videorope_exp/datasets/webvid/metadata_hf_check/data/train/partitions/0322.csv
      # expected：videoid,contentUrl,duration,page_dir,name
    ```

  ## download metadata and construct sqlite
    ```bash
      mkdir -p videorope_exp/datasets/webvid/metadata_hf_full
      hf download TempoFunk/webvid-10M \
        --repo-type dataset \
        --local-dir videorope_exp/datasets/webvid/metadata_hf_full \
        --include "data/train/partitions/*.csv"

      python videorope_exp/tools/videoweave/00_build_webvid_url_db.py
    ```

  ## check sqlite and sample debug / 10k manifest
    ```bash
      python videorope_exp/tools/videoweave/00_inspect_webvid_db.py \
        --db videorope_exp/datasets/webvid/metadata/webvid_url.db \
        --table videos \
        --limit 10 \
        --report videorope_exp/datasets/webvid/metadata/inspect_report_webvid_url.json

      mkdir -p videorope_exp/datasets/webvid/subsets/webvid_100_debug_seed42_url
      python videorope_exp/tools/videoweave/01_sample_webvid_subset.py \
        --db videorope_exp/datasets/webvid/metadata/webvid_url.db \
        --table videos \
        --size 100 \
        --seed 42 \
        --id-col videoid \
        --caption-col name \
        --url-col contentUrl \
        --outdir videorope_exp/datasets/webvid/subsets/webvid_100_debug_seed42_url

      mkdir -p videorope_exp/datasets/webvid/subsets/webvid_10k_seed42_url
      python videorope_exp/tools/videoweave/01_sample_webvid_subset.py \
        --db videorope_exp/datasets/webvid/metadata/webvid_url.db \
        --table videos \
        --size 10000 \
        --seed 42 \
        --id-col videoid \
        --caption-col name \
        --url-col contentUrl \
        --outdir videorope_exp/datasets/webvid/subsets/webvid_10k_seed42_url
    ```

  ## download video
    ```bash
      # debug
      conda activate download
      python videorope_exp/tools/videoweave/02_download_videos_sequential.py \
        --manifest videorope_exp/datasets/webvid/subsets/webvid_100_debug_seed42_url/download_manifest.csv \
        --outdir videorope_exp/datasets/webvid/videos_seq/webvid_20_debug_seed42_url \
        --max-items 20 \
        --timeout 30 \
        --retries 3 \
        --sleep 1.0 \
        --skip-existing

      # 10k download
      bash videorope_exp/tools/videoweave/run_download_10k_seq.sh
      # 输出：videorope_exp/datasets/webvid/videos_seq/webvid_10k_seed42_url
    ```

  ## generate local manifest and extract frames
    ```bash
      mkdir -p videorope_exp/datasets/webvid/processed/webvid_10k_seed42_url
      python videorope_exp/tools/videoweave/03_build_local_success_manifest.py \
        --download-root videorope_exp/datasets/webvid/videos_seq/webvid_10k_seed42_url \
        --out-csv videorope_exp/datasets/webvid/processed/webvid_10k_seed42_url/local_success_manifest.csv

      mkdir -p videorope_exp/datasets/webvid/frames/webvid_10k_seed42_url_16f
      mkdir -p videorope_exp/datasets/webvid/processed/webvid_10k_seed42_url/frame_reports_full
      python videorope_exp/tools/videoweave/04_extract_frames_uniform.py \
        --manifest videorope_exp/datasets/webvid/processed/webvid_10k_seed42_url/local_success_manifest.csv \
        --frames-root videorope_exp/datasets/webvid/frames/webvid_10k_seed42_url_16f \
        --num-frames 16 \
        --resize-short 448 \
        --ext jpg \
        --skip-existing \
        --report-dir videorope_exp/datasets/webvid/processed/webvid_10k_seed42_url/frame_reports_full
    ```

  ## construct VideoWeave manifests（single / random L=2, 8+8）
    ```bash
      mkdir -p videorope_exp/datasets/webvid/processed/webvid_10k_seed42_url/manifests
      python videorope_exp/tools/videoweave/05_build_videoweave_manifests.py \
        --local-manifest videorope_exp/datasets/webvid/processed/webvid_10k_seed42_url/local_success_manifest.csv \
        --frames-root videorope_exp/datasets/webvid/frames/webvid_10k_seed42_url_16f \
        --outdir videorope_exp/datasets/webvid/processed/webvid_10k_seed42_url/manifests \
        --seed 42 \
        --single-frames 16 \
        --videos-per-sample 2 \
        --frames-per-video 8 \
        --image-ext jpg
      # output：single_video_16f_manifest.jsonl / videoweave_random_l2_f8_manifest.jsonl
    ```

  ## render video and generate LLaMA-Factory / VideoRoPE training dataset
    ```bash
      mkdir -p videorope_exp/datasets/webvid/rendered_videos_full
      python videorope_exp/tools/videoweave/07_render_manifests_to_videos.py \
        --single-manifest videorope_exp/datasets/webvid/processed/webvid_10k_seed42_url/manifests/single_video_16f_manifest.jsonl \
        --videoweave-manifest videorope_exp/datasets/webvid/processed/webvid_10k_seed42_url/manifests/videoweave_random_l2_f8_manifest.jsonl \
        --out-root videorope_exp/datasets/webvid/rendered_videos_full \
        --fps 4.0 \
        --skip-existing

      mkdir -p videorope_exp/datasets/llamafactory_webvid_video_full
      python videorope_exp/tools/videoweave/08_build_llamafactory_video_datasets.py \
        --single-rendered videorope_exp/datasets/webvid/rendered_videos_full/single_video_16f/rendered_manifest.jsonl \
        --videoweave-rendered videorope_exp/datasets/webvid/rendered_videos_full/videoweave_random_l2_f8/rendered_manifest.jsonl \
        --dataset-dir videorope_exp/datasets/llamafactory_webvid_video_full \
        --single-name webvid_single_16f_video_sharegpt \
        --videoweave-name webvid_videoweave_l2_f8_video_sharegpt \
        --pilot-size 512 \
        --seed 42
    ```

  ## training datasets VideoRoPE / LLaMA-Factory
    ```text
      webvid_single_16f_video_sharegpt
      webvid_videoweave_l2_f8_video_sharegpt
      webvid_single_16f_video_sharegpt_pilot512
      webvid_videoweave_l2_f8_video_sharegpt_pilot512
    ```

