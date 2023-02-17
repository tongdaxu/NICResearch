```bash
/home/JJ_Group/yinsz/data/CLIC2020/

python train.py -u -d /home/JJ_Group/yinsz/data/CLIC2020/ --epochs 3000 -lr 1e-4 --batch-size 16 --cuda --save
```

python -u test_perceptual.py -d /home/JJ_Group/xutd/git/Dataset/COCO --epochs 128 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0003 --cuda --save --checkpoint  ./37012/ckp_30.pth


python -u train_con_perceptual.py --epochs 128 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0003 --cuda --save