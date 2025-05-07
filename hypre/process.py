import re
import os
import glob
import pandas as pd

def extract_data(filename):
    """Extracts data from a log file."""
    try:
        with open(filename, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        return None  # Handle missing files gracefully

    # Use regular expressions to extract data.  Adjust these patterns if needed!
    setup_time_match = re.search(r"PCG Setup:\s*wall clock time = (\d+\.\d+)", log_content)
    solve_time_match = re.search(r"PCG Solve:\s*wall clock time = (\d+\.\d+)", log_content)
    iterations_match = re.search(r"Iterations = (\d+)", log_content)
    residual_match = re.search(r"Final Relative Residual Norm = (\d+\.\d+e[-+]\d+)", log_content)

    if (setup_time_match and solve_time_match and iterations_match and residual_match):
        setup_time = float(setup_time_match.group(1))
        solve_time = float(solve_time_match.group(1))
        iterations = int(iterations_match.group(1))
        residual = float(residual_match.group(1))
        return setup_time, solve_time, iterations, residual
    else:
        return None


# Get a list of all log files in the current directory
log_files = glob.glob("*.log")

data = []
for log_file in log_files:
    match = re.match(r"([\w-]+)_(\d+)\.log", log_file)
    if match:
        matrix_name = match.group(1)
        num_procs = int(match.group(2))  # Convert to integer for sorting

        extracted_data = extract_data(log_file)
        if extracted_data:
            setup_time, solve_time, iterations, residual = extracted_data
            data.append([matrix_name, num_procs, setup_time, solve_time, iterations, residual])
        else:
            print(f"Error processing {log_file}: Could not extract all data.")
    else:
        print(f"Skipping file {log_file}: Does not match naming convention.")


# Create a Pandas DataFrame
df = pd.DataFrame(data, columns=['Matrix Name', 'Processes', 'Setup Time', 'Solve Time', 'Iterations', 'Residual'])

# Sort the DataFrame
df = df.sort_values(['Matrix Name', 'Processes'])

# Add 't' to the Processes column for display
df['Processes'] = df['Processes'].astype(str) + 't'

# Display or save the DataFrame
print(df)

#Optional: Save to CSV
df.to_csv('results.csv', index=False)