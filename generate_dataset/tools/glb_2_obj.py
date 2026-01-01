import numpy as np
from pygltflib import GLTF2, BufferFormat
import base64

# A mapping from accessor type to the number of components
ACCESSOR_TYPE_NUM_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

def load_buffer_data(buffer, gltf_file):
    if buffer.uri:
        if buffer.uri.startswith('data:'):
            # Embedded base64 encoded data
            data = base64.b64decode(buffer.uri.split(',')[1])
        else:
            # External .bin file
            bin_file = gltf_file[:gltf_file.rfind('/') + 1] + buffer.uri
            with open(bin_file, 'rb') as f:
                data = f.read()
    else:
        # Binary data is embedded in the .glb file
        data = buffer.data

    return data

def extract_vertices_and_faces(gltf_file):
    gltf = GLTF2().load(gltf_file)

    vertices = []
    faces = []

    for buffer in gltf.buffers:
        buffer.data = load_buffer_data(buffer, gltf_file)

    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            # Extract vertices
            accessor = gltf.accessors[primitive.attributes.POSITION]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            buffer = gltf.buffers[buffer_view.buffer]

            byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
            num_components = ACCESSOR_TYPE_NUM_COMPONENTS[accessor.type]
            byte_length = accessor.count * num_components * 4  # assuming float32
            data = np.frombuffer(buffer.data[byte_offset:byte_offset + byte_length], dtype=np.float32)

            positions = data.reshape(accessor.count, num_components)
            vertices.extend(positions)

            # Extract faces (indices)
            if primitive.indices is not None:
                index_accessor = gltf.accessors[primitive.indices]
                index_buffer_view = gltf.bufferViews[index_accessor.bufferView]
                index_buffer = gltf.buffers[index_buffer_view.buffer]

                index_byte_offset = (index_buffer_view.byteOffset or 0) + (index_accessor.byteOffset or 0)
                index_byte_length = index_accessor.count * index_accessor.ComponentType.size

                if index_accessor.componentType == index_accessor.ComponentType.UNSIGNED_SHORT:
                    indices = np.frombuffer(index_buffer.data[index_byte_offset:index_byte_offset + index_byte_length], dtype=np.uint16)
                elif index_accessor.componentType == index_accessor.ComponentType.UNSIGNED_INT:
                    indices = np.frombuffer(index_buffer.data[index_byte_offset:index_byte_offset + index_byte_length], dtype=np.uint32)
                else:
                    raise ValueError("Unsupported index component type")

                for i in range(0, len(indices), 3):
                    faces.append((indices[i], indices[i + 1], indices[i + 2]))

    return vertices, faces

def write_obj(vertices, faces, obj_file):
    with open(obj_file, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

def convert_glb_to_obj(glb_file, obj_file):
    vertices, faces = extract_vertices_and_faces(glb_file)
    write_obj(vertices, faces, obj_file)

def main():
    glb_file = '/opt/data/private/code/NormNet/generate_dataset/models/apple.glb'  # 替换为你的 .glb 文件路径
    obj_file = '/opt/data/private/code/NormNet/generate_dataset/models/apple.obj'  # 替换为你想要输出的 .obj 文件路径
    convert_glb_to_obj(glb_file, obj_file)
    print(f"Converted {glb_file} to {obj_file}")

if __name__ == "__main__":
    main()
