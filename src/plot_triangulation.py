import plotly.graph_objects as go
import numpy as np
import json
import sys

def parse_data_file(filepath):
    """
    Parses a JSON file containing a list of simplices: triangles in 2D
    (3 vertices with 2 coordinates each) or tetrahedra in 3D (4 vertices
    with 3 coordinates each).
    e.g., [[[x,y], [x,y], [x,y]], ...] or [[[x,y,z], [x,y,z], [x,y,z], [x,y,z]], ...]
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Convert the list of lists to a list of numpy arrays
        return [np.array(tetra) for tetra in data]
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from '{filepath}'. The file may be empty or malformed.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        return None


def scale_simplex(vertices, scale_factor=0.8):
    """
    Scales a simplex's vertices towards its centroid to improve visibility.
    """
    centroid = np.mean(vertices, axis=0)
    scaled_vertices = centroid + scale_factor * (vertices - centroid)
    return scaled_vertices

def plot_triangulation(tetrahedra_vertices):
    """
    Creates a generalized interactive 3D plot for a list of tetrahedra.
    The plot boundaries are calculated automatically.
    """
    if not tetrahedra_vertices:
        print("No simplices to plot.")
        return

    if tetrahedra_vertices[0].shape[1] == 2:
        plot_triangulation_2d(tetrahedra_vertices)
        return

    i_faces = [0, 0, 0, 1]
    j_faces = [1, 1, 2, 2]
    k_faces = [2, 3, 3, 3]

    fig = go.Figure()

    for i, vertices in enumerate(tetrahedra_vertices):
        scaled_verts = scale_simplex(vertices)
        fig.add_trace(go.Mesh3d(
            x=scaled_verts[:, 0],
            y=scaled_verts[:, 1],
            z=scaled_verts[:, 2],
            i=i_faces,
            j=j_faces,
            k=k_faces,
            opacity=1,
            name=f'Tetrahedron {i+1}',
            flatshading=True
        ))
        
    all_vertices = np.vstack(tetrahedra_vertices)
    
    min_coords = np.min(all_vertices, axis=0)
    max_coords = np.max(all_vertices, axis=0)
    
    margin = (max_coords - min_coords).max() * 0.1
    
    fig.update_layout(
        title_text='Interactive Triangulation from File',
        scene=dict(
            xaxis=dict(title='X', range=[min_coords[0] - margin, max_coords[0] + margin]),
            yaxis=dict(title='Y', range=[min_coords[1] - margin, max_coords[1] + margin]),
            zaxis=dict(title='Z', range=[min_coords[2] - margin, max_coords[2] + margin]),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

def plot_triangulation_2d(triangles):
    """
    Creates an interactive 2D plot for a list of triangles.
    """
    fig = go.Figure()

    for i, vertices in enumerate(triangles):
        scaled_verts = scale_simplex(vertices)
        closed = np.vstack([scaled_verts, scaled_verts[:1]])
        fig.add_trace(go.Scatter(
            x=closed[:, 0],
            y=closed[:, 1],
            fill='toself',
            mode='lines',
            name=f'Triangle {i+1}'
        ))

    fig.update_layout(
        title_text='Interactive Triangulation from File',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y', scaleanchor='x', scaleratio=1),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        print("Usage: python plot_triangulation.py <path_to_data_file>")
        print("Example: python plot_triangulation.py temp_triangulation.json")
        sys.exit(1)

    triangulation = parse_data_file(input_file)

    if triangulation:
        plot_triangulation(triangulation)
    else:
        print(f"Could not generate plot from '{input_file}' due to parsing errors or lack of data.")
