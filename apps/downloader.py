import gradio as gr
import os
import shutil
import tempfile

# Define the root directory you want to expose. 
# "." means the directory where the script is running. 
# Change this to an absolute path (e.g., "/var/www/html" or "C:/Users/Data") if needed.
BASE_DIR = "."

def prepare_download(selected_path):
    """
    Checks the selected path. Returns a file directly, 
    or zips a directory into a temporary folder and returns the zip.
    """
    if not selected_path:
        raise gr.Error("Please select a file or directory first.")
    
    # Ensure the path is absolute
    full_path = os.path.abspath(selected_path)
    
    if not os.path.exists(full_path):
        raise gr.Error("File or directory not found on the server.")
    
    # CASE 1: It's a file -> Send it straight to the download component
    if os.path.isfile(full_path):
        return full_path
        
    # CASE 2: It's a directory -> Zip it first
    elif os.path.isdir(full_path):
        # Create a temporary directory to hold the zip file so we don't clutter the server
        temp_dir = tempfile.mkdtemp()
        
        # Get the name of the folder to name the zip file accordingly
        folder_name = os.path.basename(full_path) or "server_archive"
        zip_base_path = os.path.join(temp_dir, folder_name)
        
        # Create the zip archive (shutil automatically appends the '.zip' extension)
        try:
            archive_path = shutil.make_archive(
                base_name=zip_base_path, 
                format='zip', 
                root_dir=full_path
            )
            return archive_path
        except Exception as e:
            raise gr.Error(f"Failed to zip directory: {str(e)}")
    else:
        raise gr.Error("Invalid selection.")

# Build the Gradio UI
with gr.Blocks(title="Server File Downloader") as demo:
    gr.Markdown("# 🗄️ Server File & Directory Downloader")
    gr.Markdown("Navigate the server filesystem. Select a file to download it directly, or select a folder to download its contents as a `.zip` archive.")
    
    with gr.Row():
        # FileExplorer requires Gradio 4.0+
        file_explorer = gr.FileExplorer(
            root_dir=BASE_DIR,
            ignore_glob=".*", # Ignores hidden files/folders (like .git) for safety
            file_count="single",
            label="Navigate Server Filesystem"
        )
    
    with gr.Row():
        download_btn = gr.Button("Prepare Download", variant="primary")
        
    with gr.Row():
        # The output component where the prepared file/zip will appear for download
        download_output = gr.File(label="Download Ready", interactive=False)
        
    # Wire the button click to the backend function
    download_btn.click(
        fn=prepare_download,
        inputs=file_explorer,
        outputs=download_output
    )

if __name__ == "__main__":
    # Launch the server. 
    # server_name="0.0.0.0" exposes it to your local network.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)