import numpy as np

def read_obj_file(file_path):
    vertices = []
    other_lines = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex position data
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            else:
                other_lines.append(line)
    
    return np.array(vertices), other_lines

def scale_vertices(vertices, scale_factor):
    return vertices * scale_factor

def write_obj_file(file_path, vertices, other_lines):
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for line in other_lines:
            file.write(line)

# Example usage:
file_path = '/opt/data/private/code/NormNet/generate_dataset/models/normalized_object/milk_box.obj'
output_path = '/opt/data/private/code/NormNet/generate_dataset/models/milk_box_2.obj'
scale_factor = 108  # Replace with your desired scale factor

vertices, other_lines = read_obj_file(file_path)
scaled_vertices = scale_vertices(vertices, scale_factor)
write_obj_file(output_path, scaled_vertices, other_lines)

print(f"Scaled .obj file saved to {output_path}")
