import os
import sys
os.system('python test.py \
--dataroot \
../datasets/SICE/train \
--no_dropout \
--name \
enlightening \
--dataset_mode \
pair \
--which_model_netG \
sid_unet_resize \
--which_model_netD \
no_norm_4 \
--n_layers_D \
5 \
--n_layers_patchD \
4 \
--fineSize \
320 \
--patchSize \
32 \
--skip \
1 \
--batchSize \
2 \
--use_norm \
1 \
--use_wgan \
0 \
--use_ragan \
--hybrid_loss \
--times_residual \
--lr \
0.00002 \
--continue_train \
0 \
--continue_epoch \
1 \
--fullinput \
0 \
--patchD_3 \
4 \
--hasglobal \
1 \
--instance_norm \
0 \
--vgg \
0 \
--vgg_choose \
relu5_1 \
--gpu_ids \
0 \
--display_port \
8097 \
--ssim_loss \
0.2 \
--self_attention \
--model \
separate \
--retina \
1 \
'
)
