from utils.utils import FPS_sample, read_ply

if __name__ == '__main__':
    ply_file_path = "/opt/data/private/code/NormNet/real_result/result_point_cloud.ply"
    point_clouds = read_ply(ply_file_path)