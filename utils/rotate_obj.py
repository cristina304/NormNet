import numpy as np

def rotate_around_z(vertices, angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)
    
    # Define the rotation matrix for 180 degrees around Z-axis
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians),  np.cos(angle_radians), 0],
        [0,                     0,                     1]
    ])
    
    # Rotate each vertex
    rotated_vertices = []
    for vertex in vertices:
        rotated_vertex = np.dot(rotation_matrix, vertex)
        rotated_vertices.append(rotated_vertex)
    
    return rotated_vertices

def read_obj(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                vertices.append(vertex)
    return vertices, lines

def write_obj(file_path, lines, rotated_vertices):
    with open(file_path, 'w') as file:
        vertex_index = 0
        for line in lines:
            if line.startswith('v '):
                v = rotated_vertices[vertex_index]
                file.write(f"v {v[0]} {v[1]} {v[2]}\n")
                vertex_index += 1
            else:
                file.write(line)

def main():
    input_file = '/opt/data/private/code/NormNet/grasp/grasper_model/three_finger_grasper.obj'
    output_file = '/opt/data/private/code/NormNet/grasp/grasper_model/three_finger_grasper_Zrotated180.obj'
    
    vertices, lines = read_obj(input_file)
    rotated_vertices = rotate_around_z(vertices, 180)
    write_obj(output_file, lines, rotated_vertices)

if __name__ == "__main__":
    main()
