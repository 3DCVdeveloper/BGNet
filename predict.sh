# python predict.py --dataset 'kitti_12' \
                  # --model 'bgnet' \
                  # --datapath '/data/home/user_name/dataset/dataset/data_stereo_flow/' \
                  # --testlist './filenames/KITTI-12-Test.txt' \
                  # --savepath './kitti-12-BGNet/' \
                  # --resume './models/kitti_12_BGNet.pth'
# exit
# python predict.py --dataset 'kitti_12' \
                  # --model 'bgnet_plus' \
                  # --datapath '/data/home/user_name/dataset/dataset/data_stereo_flow/' \
                  # --testlist './filenames/KITTI-12-Test.txt' \
                  # --savepath './kitti-12-BGNet-Plus/' \
                  # --resume './models/kitti_12_BGNet_Plus.pth'
python predict.py --dataset 'kitti' \
                  --model 'bgnet_plus' \
                  --datapath '/data/home/user_name/dataset/dataset/kitti/' \
                  --testlist './filenames/KITTI-15-Test.txt' \
                  --savepath './kitti-15-BGNet-Plus/' \
                  --resume './models/kitti_15_BGNet_Plus.pth'
#python predict.py --dataset 'kitti' \
                  #--model 'bgnet' \
                  #--datapath '/data/home/user_name/dataset/dataset/kitti/' \
                  #--testlist './filenames/KITTI-15-Test.txt' \
                  #--savepath './kitti-15-BGNet/' \
                  #--resume './models/kitti_15_BGNet.pth'