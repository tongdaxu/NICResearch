## gg18 original
* 30223_4294967294

## gg18 delta ex
* plain
* 30569_4294967294
* 
* detach noisy delta
* 30592_4294967294
* fails, reconstruction loss is very large

* bound 0.5, 1.5
* 30676
* bound 0.5 1.5 c=192
* 30753
* bound 0.8 1.25
* 30758

# Test on COOC with pretrained from CompressAI
* quality = 1
  * Bpp loss: 0.15 |	PSNR val: 25.25
* quality = 2
* quality = 3 
  * Bpp loss: 0.38 |	PSNR val: 28.17
* quality = 4
  * Bpp loss: 0.57 |	PSNR val: 29.82
* quality = 5 
  * Bpp loss: 0.78 |	PSNR val: 30.85
* quality = 6 
  * Bpp loss: 1.17 |	PSNR val: 33.58
* quality = 7 
  * Bpp loss: 1.57 |	PSNR val: 35.01
* quality = 8 

# gg18 scale hyper
# train on COCO
* lam 0.01
* patch 256: 36794_4294967294.out
* MSE loss: 0.001 |	Bpp loss: 0.52 |	PSNR val: 29.97
* lam 0.0025 36864_4294967294.out
* MSE loss: 0.002 |	Bpp loss: 0.21 |	PSNR val: 26.30
* lam 0.0005 36944_4294967294.out
* lam 0.0001 36945_4294967294.out

# Perceptual on COOC
* lam 0.0001 + 0.005 36953_4294967294.out


# gg18 mean scale hyper
* lam 0.0001 + 0.005 
  * Adam 36983_4294967294.out
  * RMSprop 36992_4294967294.out
* lam 0.0001 + 0.0001
  * 37005_4294967294.out
  * bpp too low
* lam 0.0003 + 0.0001
  * 37012_4294967294.out


# con
* lam 0.003 + 0.0001
  * 37067_4294967294.out

# net of yan ze yu
* lam 0.0003
  * 37108_4294967294.out
* lam 0.0003 stochastic gan
  * 37114_4294967294.out
* lam 0.025
  * 37107_4294967294.out
* lam 0.025 stochastic gen
  * 37112_4294967294.out

# Basemodel gg18 mean scale
* lam 0.0003
  * 2023-02-23-14-00-00
  * Average losses:	Loss: 0.099574 |	MSE loss: 0.004065 |	Bpp loss: 0.020279 |	PSNR val: 23.909569 |	Aux loss: 22.131746
* lam 0.0006
  * 2023-02-23-02-00-37
  * Average losses:	Loss: 0.149172 |	MSE loss: 0.002926 |	Bpp loss: 0.035018 |	PSNR val: 25.337400 |	Aux loss: 18.554039
* lam 0.00125
  * 2023-02-22-12-20-19
  * Average losses:	Loss: 0.215927 |	MSE loss: 0.001861 |	Bpp loss: 0.064648 |	PSNR val: 27.302095 |	Aux loss: 15.899045
* lam 0.0025
  * 2023-02-22-07-43-11
  * Average losses:	Loss: 0.298758 |	MSE loss: 0.001156 |	Bpp loss: 0.110862 |	PSNR val: 29.371021 |	Aux loss: 11.719555

# Conmodel gg18 mean scale
* lam 0.0003
  * 2023-02-23-13-58-58
  * Average losses:	Loss: 0.094564 |	MSE loss: 0.003871 |	Bpp loss: 0.019048 |	PSNR val: 24.121592 |	Aux loss: 14.357903
* lam 0.0006
  * 2023-02-23-02-00-22
  * Average losses:	Loss: 0.143894 |	MSE loss: 0.002844 |	Bpp loss: 0.032932 |	PSNR val: 25.460579 |	Aux loss: 13.006003
* lam 0.00125
  * 2023-02-22-12-19-47
  * Average losses:	Loss: 0.211611 |	MSE loss: 0.001845 |	Bpp loss: 0.061627 |	PSNR val: 27.339457 |	Aux loss: 12.879738
* lam 0.0025
  * 2023-02-22-07-08-38
  * Average losses:	Loss: 0.295381 |	MSE loss: 0.001159 |	Bpp loss: 0.107006 |	PSNR val: 29.359966 |	Aux loss: 9.544591

python -u train.py -d /data/xutd/cityspace --epochs 1000 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0025 --cuda --save

nohup python -u train_con.py -d /data/xutd/cityspace --epochs 1000 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0003 --cuda --save &> lam0003con.out &

nohup python -u train.py -d /data/xutd/cityspace --epochs 1000 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0003 --cuda --save &> lam0003.out &

nohup python -u train.py -d /data/xutd/cityspace --epochs 1000 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0006 --cuda --save &> lam0006.out &

python -u train.py --name oasis_cityscapes --dataset_mode cityscapes --gpu_ids -1 \
--dataroot /home/JJ_Group/xutd/git/Dataset/cityspace --batch_size 20 --no_EMA --no_3dnoise --no_labelmix

python -u train.py --name oasis_cityscapes --dataset_mode cityscapes --gpu_ids 0 --dataroot /data/xutd/cityspace --batch_size 20 --no_EMA --no_3dnoise --no_labelmix


python -u test.py -d /data/xutd/cityspace --lambda 0.0003 --cuda --checkpoint=2023-02-23-13-58-58/ckp_999.pth

nohup python -u train_con.py -d /data/xutd/cityspace --epochs 1000 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0006 --cuda --save &> lam0006d8.out &

# Downsample 8 times
* lam 0.00015
  * nohup python -u train_con.py -d /data/xutd/cityspace --epochs 1000 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.00015 --cuda --save &> lam00015d8.out &
  * 2023-02-27-09-11-34
* lam 0.0003
  * nohup python -u train_con.py -d /data/xutd/cityspace --epochs 1000 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0003 --cuda --save &> lam0006d3.out &
  * 2023-02-27-07-35-05
* lam 0.0006
  * nohup python -u train_con.py -d /data/xutd/cityspace --epochs 1000 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0006 --cuda --save &> lam0006d8.out &
  * 2023-02-27-07-36-53
* lam 0.00125
  * nohup python -u train_con.py -d /data/xutd/cityspace --epochs 1000 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.00125 --cuda --save &> lam00125d8.out &
  * 2023-02-27-09-12-03

* lam 0.0025
