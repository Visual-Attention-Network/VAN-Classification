MODEL=van_tiny # van_{tiny, small, base, large}
python3 validate.py /path/to/imagenet --model $MODEL \
  --checkpoint /path/to/model -b 128

