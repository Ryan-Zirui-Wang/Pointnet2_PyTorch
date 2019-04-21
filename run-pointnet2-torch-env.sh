docker run -it --rm --runtime=nvidia \
    -p 8888:8888 \
    -v /home/ryan/repos-developing/Pointnet2_PyTorch:/home/ryan/repos-developing/Pointnet2_PyTorch \
    -v /media/ryan/hdd:/media/ryan/hdd \
    -v /home/ryan/Desktop/PointClouds:/home/ryan/Desktop/PointClouds \
    ziruiryanwang/pointnet2-torch


# docker run -it --rm --runtime=nvidia \
#     -v /home/ryan/repos-developing/Pointnet2_PyTorch:/home/ryan/repos-developing/Pointnet2_PyTorch \
#     -v /media/ryan/hdd:/media/ryan/hdd \
#     ziruiryanwang/pointnet2-torch


# docker run -it --rm --runtime=nvidia \
#     -p 8888:8888 \
#     -v /home/ryan/repos-developing/Pointnet2_PyTorch:/home/ryan/repos-developing/Pointnet2_PyTorch \
#     -v /media/ryan/hdd:/media/ryan/hdd \
#     ziruiryanwang/pointnet2-torch:my-etw