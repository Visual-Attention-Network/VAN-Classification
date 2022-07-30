MODEL=van_b0 # van_{b0, b1, b2, b3}
DROP_PATH=0.1 # drop path rates [0.1, 0.1, 0.1, 0.2] for [b0, b1, b2, b3]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash distributed_train.sh 8 /path/to/imagenet \
	  --model $MODEL -b 128 --lr 1e-3 --drop-path $DROP_PATH
