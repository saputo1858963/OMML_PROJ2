import os

# --- Configuration ---
# The base directory where the Q_i folders and files will be created.
# NOTE: Ensure this path exists on your system.
BASE_DIR = r"C:\Users\Matti\OneDrive\Documenti\GitHub\OMML_PROJ2"

# The mandatory group name from the instructions.
GROUP_NAME = "BIAtomici"

# The list of question numbers (i=1, 2, 3, 4) for which files are needed.
QUESTION_NUMBERS = [1, 2, 3, 4]

# --- Folder and File Creation Logic ---

print(f"Starting creation of project structure for group: {GROUP_NAME}")
print(f"Targeting base directory: {BASE_DIR}\n")

# Check if the base directory exists
if not os.path.isdir(BASE_DIR):
    print(f"ERROR: The specified base directory does not exist: {BASE_DIR}")
    print("Please ensure the path is correct before running the script.")
else:
    # Iterate through the question numbers to create folders and files
    for i in QUESTION_NUMBERS:
        # 1. Define folder path
        folder_name = f"Q_{i}"
        full_path = os.path.join(BASE_DIR, folder_name)
        
        # 2. Define the two required file names
        # Example: run.1_BIAtomici.py
        run_file_name = f"run_{i}_{GROUP_NAME}.py"
        # Example: functions_1_BIAtomici.py
        functions_file_name = f"functions_{i}_{GROUP_NAME}.py"
        
        try:
            # Create the folder (if it doesn't already exist)
            os.makedirs(full_path, exist_ok=True)
            print(f"--> Directory created/verified: {folder_name}")
            
            # Create the 'run' file (main execution script)
            run_file_path = os.path.join(full_path, run_file_name)
            with open(run_file_path, 'w') as f:
                # Add a brief comment to make it a valid, non-empty Python file
                f.write(f"# Main script for Question {i} - Group {GROUP_NAME}\n\n")
                
            print(f"    - Created file: {run_file_name}")

            # Create the 'functions' file (helper functions)
            functions_file_path = os.path.join(full_path, functions_file_name)
            with open(functions_file_path, 'w') as f:
                # Add a brief comment
                f.write(f"# Helper functions for Question {i} - Group {GROUP_NAME}\n\n")
                
            print(f"    - Created file: {functions_file_name}\n")

        except OSError as e:
            print(f"ERROR: Could not create structure for {folder_name}. Reason: {e}\n")

    print("Project file structure successfully generated!")