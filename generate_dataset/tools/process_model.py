import numpy as np
from pygltflib import GLTF2

def calculate_bounding_box(gltf_file):
    gltf = GLTF2().load(gltf_file)
    
    min_corner = np.array([float('inf'), float('inf'), float('inf')])
    max_corner = np.array([-float('inf'), -float('inf'), -float('inf')])

    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            accessor = gltf.accessors[primitive.attributes.POSITION]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            buffer = gltf.buffers[buffer_view.buffer]

            byte_offset = buffer_view.byteOffset + accessor.byteOffset
            byte_length = accessor.count * accessor.type.value * 4  # assuming float32
            data = np.frombuffer(buffer.data[byte_offset:byte_offset + byte_length], dtype=np.float32)

            positions = data.reshape(accessor.count, 3)
            min_corner = np.minimum(min_corner, positions.min(axis=0))
            max_corner = np.maximum(max_corner, positions.max(axis=0))

    return min_corner, max_corner

def main():
    glb_file = '/opt/data/private/code/NormNet/generate_dataset/models/apple.glb'  # 替换为你的 .glb 文件路径
    min_corner, max_corner = calculate_bounding_box(glb_file)
    print(f"Bounding Box Min Corner: {min_corner}")
    print(f"Bounding Box Max Corner: {max_corner}")

if __name__ == "__main__":
    main()
