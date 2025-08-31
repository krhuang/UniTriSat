import os

def generate_config_file(index):
    """
    Generates a single .toml configuration file based on a template.

    Args:
        index (int): The number to use for the filename (vx.toml) and
                     the Polytopes file path (Polytopes/x).
        output_dir (str): The directory where the config file will be saved.
    """
    # This is the template for the .toml file.
    # The {index} placeholder will be replaced with the provided number.
    config_content = f'''
[files]
polytopes_file = "Polytopes/small_lattice_polytopes/data/5-polytopes/v{index}.txt"
log_file = ""

[run_settings]
process_range = "-"      # A space-separated list of indices and ranges (e.g., "1-5 8 10-"). "-" means all.
processing_order = "normal"    # Options: "normal", "reversed", "random"
sort_by = "none"            # "none": Use the order from the file. "P": Sort by # lattice points. "S": Sort by # simplices.
solve_mode = "count_only"         # "first": Stop after one solution. "all": Find all solutions. "count_only": Count all solutions.
find_all_simplices = false     # If true, finds all non-degenerate simplices. If false, only unimodular ones.
intersection_backend = "gpu_rationals"    # Options: "cpu", "gpu_rationals", "gpu_floats"

[output_levels]
# Options for terminal_output:
# "verbose": Detailed, multi-line output for each step.
# "multi-line": A minimal summary line is printed for each completed polytope.
# "single-line": A summary block is updated in-place in the terminal.
# "none": No output to the terminal.
terminal_output = "single-line"

# Options for file_output: "verbose", "minimal", "none"
file_output = "none"

[verbose_options]
# These toggles only apply when an output level is set to "verbose".
show_initial_vertices = true
show_solution_simplices = true
show_timing_summary = true

[solver_options]
solver = "PicoSAT" # The SAT solver to use. PicoSAT or CryptoMiniSat.

[plotting]
plotter_script = "plot_triangulation.py"
plot_range = "" # Same format as process_range. Defines which solutions to plot.
'''

    # Write the content to the file.
    file_path = f"5d/v{index}_5d_count.toml"
    try:
        with open(file_path, 'w') as f:
            f.write(config_content)
        print(f"Successfully generated '{file_path}'")
    except IOError as e:
        print(f"Error writing to file '{file_path}': {e}")


def main():
    """
    Main function to get user input and generate the files.
    """
    print("--- Config File Generator ---")
    while True:
        try:
            # Get the desired range from the user.
            start_str = input("Enter the starting number for the range: ")
            start_range = int(start_str)

            end_str = input("Enter the ending number for the range: ")
            end_range = int(end_str)

            if start_range > end_range:
                print("Error: The starting number cannot be greater than the ending number. Please try again.")
                continue

            # Generate a file for each number in the specified range (inclusive).
            for i in range(start_range, end_range + 1):
                generate_config_file(i)

            print("\nGeneration complete.")
            break # Exit the loop after successful generation

        except ValueError:
            # Handle cases where the user enters non-numeric input.
            print("Invalid input. Please enter whole numbers for the range.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    main()

