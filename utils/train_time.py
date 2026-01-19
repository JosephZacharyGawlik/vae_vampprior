import os
from datetime import datetime

def calculate_and_save_train_times(base_dir='snapshots'):
    # Get all snapshot directories
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for d in dirs:
        model_path = os.path.join(base_dir, d, 'vae.model')
        eval_dir = os.path.join(base_dir, d, 'eval_results')
        
        if not os.path.exists(model_path):
            continue
            
        try:
            # 1. Get Start Time from folder name (Format: 2026-01-17 21:55:14_...)
            # We take the first 19 characters for the date and time
            start_str = d[:19]
            start_dt = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
            
            # 2. Get End Time from the file creation/modification of vae.model
            end_timestamp = os.path.getmtime(model_path)
            end_dt = datetime.fromtimestamp(end_timestamp)
            
            # 3. Calculate Duration
            duration = end_dt - start_dt
            duration_seconds = duration.total_seconds()
            duration_minutes = duration_seconds / 60
            
            # 4. Ensure eval_results exists and write the file
            os.makedirs(eval_dir, exist_ok=True)
            output_file = os.path.join(eval_dir, 'train_time.txt')
            
            with open(output_file, 'w') as f:
                f.write(f"Training Start: {start_dt}\n")
                f.write(f"Training End:   {end_dt}\n")
                f.write(f"Total Seconds:  {duration_seconds:.2f}\n")
                f.write(f"Total Minutes:  {duration_minutes:.2f}\n")
            
            print(f"✅ Saved time for {d}: {duration_minutes:.2f} mins")
            
        except Exception as e:
            print(f"❌ Could not process {d}: {e}")

if __name__ == "__main__":
    calculate_and_save_train_times()
