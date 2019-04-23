from pointnet2.models.pointnet2_msg_sem import Pointnet2MSG
import torch.nn as nn

import etw_pytorch_utils.pytorch_utils as pt_utils
import torch

import torch.nn as nn
from pointnet2.models.pointnet2_msg_sem import model_fn_decorator

import numpy as np
from numpy import genfromtxt

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random

def load_model():
    # choose model
    model = Pointnet2MSG(num_classes=13, input_channels=0, use_xyz=True)
    model.cuda()

    # load model parameters from checkpoint
    checkpoint_name = "../checkpoints/poitnet2_semseg_best.pth.tar"
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint["model_state"])
    print("loaded pointnet sem seg model.")
    return model


def load_my_pt_cloud():
    # load global and live point cloud
    g = genfromtxt("/home/ryan/Desktop/PointClouds/global-using.txt", delimiter=",", dtype=np.float32)
    l = genfromtxt("/home/ryan/Desktop/PointClouds/live-using.txt", delimiter=",", dtype=np.float32)

    print("loaded global and live point cloud.")
    print("global point cloud shape: ", g.shape)
    print("live point cloud shape: ", l.shape)
    return g, l

# this currently not used.
def load_train_dataset():
    from pointnet2.data.Indoor3DSemSegLoader import Indoor3DSemSeg
    num_points = 4096
    train_set = Indoor3DSemSeg(num_points, download=False)
    return train_set

def draw_pt_cloud(pt_cloud):
    # expect pt cloud be a numpy array with shape N x 3.
    fig = pyplot.figure()
    ax = Axes3D(fig)

    x_vals = pt_cloud[:, 0]
    y_vals = pt_cloud[:, 1]
    z_vals = pt_cloud[:, 2]

    ax.scatter(x_vals, y_vals, z_vals)
    pyplot.show()

def preprocess_pt_cloud(pt_cloud_np):
    # Deal with dim, convert from np to torch, move to gpu.

    B = 1 # batch size = 1 because we only forward pass one pt cloud
    N = pt_cloud_np.shape[0] # num of pts

    pt_cloud_torch = torch.from_numpy(pt_cloud_np).unsqueeze(0).cuda()

    # we don't need label for forward processing so just all-zeros.
    labels = torch.from_numpy(np.zeros((1, B * N), dtype = int)).view(B, N).cuda()

    pt_cloud_torch_data = (pt_cloud_torch, labels)
    return pt_cloud_torch_data

def cmp_features(pt_cloud_tor, model, model_fn):
    _, _, _ = model_fn(model, pt_cloud_tor, eval=True)
    features = model.get_features()
    features = features.squeeze().transpose(1, 0)
    return features

def compute_vec_geo_feature_dist(xyz_f1, xyz_f2):
    # the f1 and f2 should has 3(xyz) + 128 elements in a row
    vec_diff = (xyz_f1 - xyz_f2)
    res = torch.pow(vec_diff, 2.0)
    geo_dist = torch.sqrt(torch.sum(res[:, :3], dim=1))
    fea_dist = torch.sqrt(torch.sum(res[:, 3:], dim=1))
    return geo_dist, fea_dist

def geo_dist_filter(geo_dist, fea_dist):
    # given a pair of points:
    #       if they are close in geometric distnace, we don' touch the feature distance.
    #       if they are far in geometric distance, we set the feature distance to inf so we can ignore 
    #          those matchings whose geo distance are large and distance is small.
    geo_filtered_fea_dist = torch.where(geo_dist < 0.02, fea_dist, torch.tensor(float('inf')).cuda())
    return geo_filtered_fea_dist

def cmp_matches(xyz_fg, xyz_fl):
    # for every point in live point cloud, find a best match in global point cloud.
    # returned matches include the 
    #       pt index in global pt cloud, 
    #       feature distance of this matching
    #       geometric distance of this matching
    matches = torch.zeros(size=(len(l), 3), dtype=torch.float32)
    for i, f1 in enumerate(xyz_fl):
        if i%5000==0: print(i)
            
        geo_dist, fea_dist = compute_vec_geo_feature_dist(f1, xyz_fg)   
        fea_dist = geo_dist_filter(geo_dist, fea_dist)
        min_fea_dist, idx = torch.min(fea_dist, dim=0)
        matches[i] = torch.tensor([idx, min_fea_dist, geo_dist[idx]])

    return matches

def cmp_vector_field(g, l, matches):
    vec = g[matches[:, 0].int().tolist()] - l

    vec_mag = matches[:, 2]
    for i, v_mag in enumerate(vec_mag):
        
        # do not show if vector is too large. this can happen when the best feature matching distance is +inf..
        if v_mag > 0.1: vec[i] = [0,0,0]
            
    return vec

if __name__ == "__main__":
    model = load_model()
    model_fn = model_fn_decorator(nn.CrossEntropyLoss())  # the model_fn do the forward pass
    g, l = load_my_pt_cloud()

    draw_pt_cloud(l)

    # pre-process my point cloud data: deal with dim, convert from np to torch, move to gpu.
    g_torch_data = preprocess_pt_cloud(g)
    l_torch_data = preprocess_pt_cloud(l)

    # get features from NN
    fg = cmp_features(g_torch_data, model, model_fn)
    fl = cmp_features(l_torch_data, model, model_fn)
    print("features for global point cloud: ", fg.shape)
    print("features for live point cloud: ", fl.shape)

    # concat xyz info with the 128-bit feature to save some computations when do geometric filtering and feature matching
    xyz_fl = torch.cat((l_torch_data[0].squeeze(), fl), dim=1)
    xyz_fg = torch.cat((g_torch_data[0].squeeze(), fg), dim=1)
    print("concat xyz and 128-bit feature for global point cloud: ", xyz_fg.shape)
    print("concat xyz and 128-bit feature for live point cloud: ", xyz_fl.shape)

    matches = cmp_matches(xyz_fg, xyz_fl)
    print(matches)

    vec = cmp_vector_field(g, l, matches)
    print("vector field shape: ", vec.shape)
    print("vector field: ", vec)

    np.savetxt(fname="/home/ryan/Desktop/PointClouds/vec.txt", X=vec, delimiter=",")
