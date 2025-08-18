"""
    process_file(input_path::String, output_path::String)

Reads a file from `input_path`, processes its content, and writes the result to `output_path`.

The processing logic is as follows:
- If a line contains a single integer, it is replaced by an empty line.
- If a line contains more than one integer (e.g., three integers separated by whitespace), it remains unchanged.
"""
function process_file(input_path::String, output_path::String)
    try
        # Open the input file for reading and the output file for writing.
        # The `do` block syntax ensures that the files are automatically closed
        # even if an error occurs.
        open(input_path, "r") do infile
            open(output_path, "w") do outfile
                # Iterate over each line in the input file.
                for line in eachline(infile)
                    # Split the line by whitespace to get an array of substrings.
                    parts = split(line)

                    # Check if the line contains only one element.
                    if length(parts) == 1
                        # If it does, write a newline character (an empty line)
                        # to the output file.
                        write(outfile, "\n")
                    else
                        # If it contains more than one element, write the
                        # original line back to the output file, followed by a newline.
                        write(outfile, line * "\n")
                    end
                end
            end
        end
        println("Successfully processed the file. Output is in '", output_path, "'.")
    catch e
        println("An error occurred: ", e)
    end
end

const input = "smooth3polytopes_50"
const output = "smooth3polytopes_50processed"
process_file(input, output)

