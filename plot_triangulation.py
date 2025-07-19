import plotly.graph_objects as go
import numpy as np
import re
import sys

def parse_data_file(filepath):
    """
    Parses a log file to extract the triangulation solution.
    It looks for the "Displaying simplices for the valid triangulation:" header
    and then parses the 4x3 matrices that follow.
    """
    tetrahedra = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None

    # --- UPDATED PARSING LOGIC ---
    # Flag to start parsing only after the specific header is found
    parsing_solution = False
    current_tetrahedron = []
    for line in lines:
        line = line.strip()

        # The header is the signal to start parsing the solution
        if "Displaying simplices for the valid triangulation:" in line:
            parsing_solution = True
            continue
        
        # Stop parsing if we hit the summary section or another major break
        if parsing_solution and line.startswith('---'):
            break

        # Ignore any lines before the solution block is found
        if not parsing_solution:
            continue

        # Skip header lines for matrices or empty/comment lines
        if not line or line.startswith('#') or 'Matrix' in line:
            # A header line signals the end of a previous tetrahedron block
            if len(current_tetrahedron) == 4:
                tetrahedra.append(np.array(current_tetrahedron))
            # Always reset for a new block
            current_tetrahedron = []
            continue

        # Parse the vertex coordinates
        parts = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(parts) == 3:
            current_tetrahedron.append([float(p) for p in parts])

    # Add the very last tetrahedron if the file ends right after it
    if len(current_tetrahedron) == 4:
        tetrahedra.append(np.array(current_tetrahedron))
    # --- END OF UPDATED LOGIC ---

    if not tetrahedra:
        print("Warning: Could not find a valid triangulation block in the file.")
        print("Ensure the file contains 'Displaying simplices for the valid triangulation:'.")
        return None

    return tetrahedra

def scale_tetrahedron(vertices, scale_factor=0.95):
    """
    Scales a tetrahedron's vertices towards its centroid to improve visibility.
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
        print("No tetrahedra to plot.")
        return
        
    # Define the 4 triangular faces for a tetrahedron using vertex indices
    i_faces = [0, 0, 0, 1]
    j_faces = [1, 1, 2, 2]
    k_faces = [2, 3, 3, 3]

    fig = go.Figure()

    # Add each scaled tetrahedron as a separate mesh
    for i, vertices in enumerate(tetrahedra_vertices):
        scaled_verts = scale_tetrahedron(vertices)
        fig.add_trace(go.Mesh3d(
            x=scaled_verts[:, 0],
            y=scaled_verts[:, 1],
            z=scaled_verts[:, 2],
            i=i_faces,
            j=j_faces,
            k=k_faces,
            opacity=0.7,
            name=f'Tetrahedron {i+1}',
            flatshading=True
        ))
        
    # --- DYNAMIC AXIS SCALING ---
    # Combine all vertices into a single NumPy array
    all_vertices = np.vstack(tetrahedra_vertices)
    
    # Find the minimum and maximum coordinates across all axes
    min_coords = np.min(all_vertices, axis=0)
    max_coords = np.max(all_vertices, axis=0)
    
    # Calculate a dynamic margin to ensure the plot is not cramped
    margin = (max_coords - min_coords).max() * 0.1 # 10% of the largest dimension
    
    # Configure the plot layout with the calculated ranges
    fig.update_layout(
        title_text='Interactive Triangulation from File',
        scene=dict(
            xaxis=dict(title='X', range=[min_coords[0] - margin, max_coords[0] + margin]),
            yaxis=dict(title='Y', range=[min_coords[1] - margin, max_coords[1] + margin]),
            zaxis=dict(title='Z', range=[min_coords[2] - margin, max_coords[2] + margin]),
            aspectmode='data' # Ensures the aspect ratio is 1:1:1
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Check if a filename is provided as a command-line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # If no argument is given, print usage instructions and exit
        print("Usage: python interactive_plot_from_file.py <path_to_log_file>")
        print("Example: python interactive_plot_from_file.py results.log")
        sys.exit(1) # Exit with an error code

    # Parse the file to get the list of tetrahedra
    triangulation = parse_data_file(input_file)

    # If data was parsed successfully, create the plot
    if triangulation:
        print(f"Successfully parsed {len(triangulation)} tetrahedra from '{input_file}'. Generating plot...")
        plot_triangulation(triangulation)
    else:
        print(f"Could not generate plot from '{input_file}' due to parsing errors or lack of data.")

