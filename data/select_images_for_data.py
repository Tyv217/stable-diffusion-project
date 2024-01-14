NAME = "Street"
PIC_DIRECTORY="C:/Users/thoma/Downloads/" + NAME + "/" + NAME
SAVE_TO_DIRECTORY="C:/Users/thoma/stable-diffusion-project/data/data_files/" + NAME

def main():
    import os
    import shutil

    # Directory containing the "seq" folders
    root_dir = PIC_DIRECTORY  # Change this to your main directory path

    # Directory where selected frames will be copied

    # Loop through each subdirectory in the root directory
    seq_dir = os.path.join(root_dir, "img_west")
    
    # Check if the directory exists
    if os.path.exists(seq_dir):
        for filename in os.listdir(seq_dir):
            # Check if the file is an image and is a multiple of 20
            filestart = "image_west"
            if filename.startswith(filestart) and int(filename[len(filestart) + 1:len(filename) - 4]) % 20 == 0:
                src_path = os.path.join(seq_dir, filename)
                dst_path = os.path.join(SAVE_TO_DIRECTORY, f'{filestart}_{filename}')
                shutil.copy(src_path, dst_path)
                print(f"Copied {src_path} to {dst_path}")
    else:
        print(f"Directory not found: {seq_dir}")

    print("Finished copying selected frames.")

if __name__ == "__main__":
    main()