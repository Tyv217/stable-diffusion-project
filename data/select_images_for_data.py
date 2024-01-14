NAME = "OldHospital"
PIC_DIRECTORY="C:/Users/thoma/Downloads/" + NAME + "/" + NAME
SAVE_TO_DIRECTORY="C:/Users/thoma/stable-diffusion-project/data/data_files/" + NAME

def main():
    import os
    import shutil

    # Directory containing the "seq" folders
    root_dir = PIC_DIRECTORY  # Change this to your main directory path

    # Directory where selected frames will be copied
    selected_dir = os.path.join(root_dir, 'selected')
    os.makedirs(selected_dir, exist_ok=True)

    # Loop through each subdirectory in the root directory
    for i in range(1, 9):
        seq_dir = os.path.join(root_dir, f'seq{i}')
        
        # Check if the directory exists
        if os.path.exists(seq_dir):
            for filename in os.listdir(seq_dir):
                # Check if the file is an image and is a multiple of 20
                if filename.startswith("frame") and int(filename[5:10]) % 20 == 0:
                    src_path = os.path.join(seq_dir, filename)
                    dst_path = os.path.join(SAVE_TO_DIRECTORY, f'seq{i}_{filename}')
                    shutil.copy(src_path, dst_path)
                    print(f"Copied {src_path} to {dst_path}")
        else:
            print(f"Directory not found: {seq_dir}")

    print("Finished copying selected frames.")

if __name__ == "__main__":
    main()