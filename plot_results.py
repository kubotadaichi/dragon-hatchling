import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_loss(log_file="results/training_log.csv", output_file="results/loss_plot.png"):
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    # Manually specify names because the header in the file is missing 'val_loss' 
    # (it wrote "epoch,step,loss,time" but data has 5 columns)
    df = pd.read_csv(log_file, header=0, names=['epoch', 'step', 'loss', 'val_loss', 'time'])
    
    # Filter out validation rows for training loss plot (loss is not NaN)
    train_df = df[pd.to_numeric(df['loss'], errors='coerce').notna()]
    
    # Filter for validation rows (val_loss is not NaN)
    val_df = df[pd.to_numeric(df['val_loss'], errors='coerce').notna()]

    plt.figure(figsize=(10, 6))
    plt.plot(train_df['step'], train_df['loss'], label='Training Loss', alpha=0.6)
    
    # Plot validation loss points
    if not val_df.empty:
        plt.plot(val_df['step'], val_df['val_loss'], 'ro-', label='Validation Loss', markersize=5)

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_loss()
