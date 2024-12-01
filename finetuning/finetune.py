import sys
from ultralytics import YOLO
from pathlib import Path
from check_gpu_availablity import get_free_gpu
from datetime import datetime
import os

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    # Ensure we're writing to absolute paths for background execution
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_filename = os.path.join(base_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = Logger(log_filename)
    
    # Log start time and process ID for tracking
    print(f"Starting training process with PID: {os.getpid()}")
    print(f"Start time: {datetime.now()}")
    
    # Get the GPU with most available memory
    gpu_id = get_free_gpu()
    device = gpu_id if gpu_id is not None else 'cpu'
    print(f"Using GPU {gpu_id} with {device}")
    
    # Load the existing fine-tuned model
    model = YOLO('yolo11x.pt')

    # Update the dataset path to use the new dataset for "puck"
    dataset_yaml = str(Path('./datasets/Player-Detection-2').absolute())

    # Start additional training for the new class
    try:
        results = model.train(
            data=f"{dataset_yaml}/data.yaml",
            task='detect',
            mode='train',
            epochs=100,
            imgsz=640,
            device=device,
            verbose=True,
            batch=-1,
        )
        
        # Save the updated modelde
        model.save('./final_model.pt')
        print("Finetuning completed successfully")
        
    except Exception as e:
        print(f"Finetuning failed with error: {e}")

if __name__ == "__main__":
    main()
