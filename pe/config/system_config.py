import os

class PoseEstimatorConfig:
    model_dir = "./pe/models"
    data_dir = "./data/"
    uploaded_dir = os.path.join(data_dir, "uploads")
    output_dir = os.path.join(data_dir, "pose_outdir")
    os.makedirs(uploaded_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    pe_model_path = os.path.join(model_dir +"/model/" , "pose_resnet_50_256x256.pth.tar")
    config_path = os.path.join(model_dir +"/config/" , "256x256_d256x3_adam_lr1e-3.yaml")
    pe_model = None
