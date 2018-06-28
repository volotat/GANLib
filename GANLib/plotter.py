import matplotlib.pyplot as plt
import numpy as np
  
def save_hist_image(hist, file):
    hist_size = hist['hist_size']

    plt.figure(figsize=(14,7))
    plt.subplot(2, 1, 1)
    plt.yscale('log')
    plt.plot(hist['metric'][:hist_size,0], '-',linewidth=0.8, label="metric mean", color='C0')
    plt.plot(hist['metric'][:hist_size,1], '-',linewidth=0.8, label="metric min", color='C0', alpha = 0.5)
    plt.plot(hist['metric'][:hist_size,2], '-',linewidth=0.8, label="metric max", color='C0', alpha = 0.5)
    plt.axhline(hist['best_metric'],linewidth=0.8)
    plt.xlabel(hist['best_metric'])
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(hist['train_val'][:hist_size,0], '-',linewidth=0.8, label="train_data_valid", color='C0')
    plt.plot(hist['test_val'][:hist_size,0], '-',linewidth=0.8, label="test_data_valid", color='C1')

    plt.plot(hist['control_val'][:hist_size,0], '-',linewidth=0.8, label="control_valid", color='C3')
    plt.plot(hist['gen_val'][:hist_size,0], '-',linewidth=0.8, label="gen_val", color='C4')

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    plt.savefig(file)
    plt.close()
    