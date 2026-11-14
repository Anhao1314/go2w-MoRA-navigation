#!/usr/bin/env bash
# Retry loop for the LIBERO dataset download: each attempt resumes from
# completed blobs; transient CDN timeouts are survived by the next attempt.
cd /mnt/d/work/go2w-locomotion
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1

for attempt in 1 2 3 4 5 6 7 8; do
  echo "=== attempt $attempt $(date -u +%H:%M:%S) ==="
  /root/vla-env/bin/python -u -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('lerobot/libero')
print('episodes:', ds.meta.total_episodes, '| frames:', ds.meta.total_frames, '| fps:', ds.meta.fps)
print('DOWNLOAD_OK tasks:', ds.meta.total_tasks)
"
  if [ $? -eq 0 ]; then
    echo "=== DOWNLOAD_COMPLETE ==="
    exit 0
  fi
  sleep 45
done
echo "=== ALL_ATTEMPTS_EXHAUSTED ==="
exit 1
