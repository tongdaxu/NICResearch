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
* lam 0.00125
* lam 0.0025
  * 37158_4294967294.out
  * Bpp loss: 0.06 |	PSNR val: 31.38

# Conmodel gg18 mean scale
* lam 0.00125
* lam 0.0025


python -u train.py --name oasis_cityscapes --dataset_mode cityscapes --gpu_ids -1 \
--dataroot /home/JJ_Group/xutd/git/Dataset/cityspace --batch_size 20 --no_EMA --no_3dnoise --no_labelmix



python -u train.py --name oasis_cityscapes --dataset_mode cityscapes --gpu_ids 0 --dataroot /data/xutd/cityspace --batch_size 20 --no_EMA --no_3dnoise --no_labelmix