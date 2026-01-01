import numpy as np

def read_obj_file(file_path):
    vertices = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex position data
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
    
    return np.array(vertices)

def compute_bounding_box(vertices):
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    return min_coords, max_coords

def bounding_box_size(min_coords, max_coords):
    return max_coords - min_coords

# Example usage:
file_path = '/opt/data/private/code/NormNet/generate_dataset/models/trapezoidal_block.obj'
vertices = read_obj_file(file_path)
min_coords, max_coords = compute_bounding_box(vertices)
size = bounding_box_size(min_coords, max_coords)

print(f"Bounding Box Min Coordinates: {min_coords}")
print(f"Bounding Box Max Coordinates: {max_coords}")
print(f"Bounding Box Size: {size}")
