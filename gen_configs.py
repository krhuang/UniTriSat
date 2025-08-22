import os

# The template for the configuration file.
# This string contains the base content for all generated files.
CONFIG_TEMPLATE = """
[files]
polytopes_file = "small-lattice-polytopes/data/4-polytopes/{filename}"
log_file = ""

[run_settings]
process_range = "-"
processing_order = "normal"			# Options: "normal", "reversed", "random"

sort_by = "none"				# "none": Use the order from the file. "P": Sort by the number of lattice points, ascending. "S": Sort by the number of simplices, ascending.
remove_origin_simplices = false
find_all_simplices = false			# If false, it is restricted to only unimodular simplices. If true, the search includes all non-degenerate simplices.
intersection_backend = "gpu"			#cpu or gpu

[output_levels]
terminal_output = "minimal" 			# Output level for the terminal. Options: "verbose", "minimal", "none"
terminal_mode = "single-line"			# Terminal output mode. "multi-line" prints a new line for each polytope. "single-line" updates a single line in place (terminal only).
file_output = "none"				# Output level for the log file. Options: "verbose", "minimal", "none"
solution_reporting = "first"    # "first": Stop after finding one solution. "all": Find all possible solutions.  "count_only": Efficiently count all solutions without storing them.

[verbose_options]
# These toggles only apply when an output level is set to "verbose".
show_initial_vertices = true
show_solution_simplices = true 			# If solution_reporting="all", this shows all solutions.
show_timing_summary = true

[solver_options]
solver = "PicoSAT"				# The SAT solver to use. PicoSAT or CryptoMiniSat, PicoSAT is default

[plotting]
plotter_script = "plot_triangulation.py"	# The python script to execute for plotting a triangulation.
plot_range = ""
"""

def generate_config_files():
    """
    Generates configuration files for a range of polytope vertex files.
    """
    # Define the range of file numbers to generate, from 4 to 33 inclusive.
    start_num = 4
    end_num = 33

    # Create a directory to store the generated files, if it doesn't exist.
    output_directory = "configs_4d"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: '{output_directory}'")

    # Loop through the specified range of numbers.
    for i in range(start_num, end_num + 1):
        # Construct the specific vertex filename (e.g., "v4.txt", "v5.txt", etc.)
        vertex_filename = f"v{i}.txt"

        # Format the template with the new vertex filename.
        # This replaces the {filename} placeholder in the template.
        config_content = CONFIG_TEMPLATE.format(filename=vertex_filename)

        # Define the name for the output configuration file.
        output_filename = os.path.join(output_directory, f"v{i}_4d.toml")

        # Write the generated content to the new file.
        try:
            with open(output_filename, "w") as f:
                f.write(config_content)
            print(f"Successfully generated '{output_filename}'")
        except IOError as e:
            print(f"Error writing to file '{output_filename}': {e}")

if __name__ == "__main__":
    generate_config_files()
    print("\nConfiguration file generation complete.")


