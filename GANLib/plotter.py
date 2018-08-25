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
    plt.xlabel('Best result: %f'%hist['best_metric'])
    plt.grid(True)

    
    plt.subplot(2, 1, 2)
    for i in range(len(hist)):
        key = list(hist.keys())[i]
        if isinstance(hist[key], np.ndarray) and key != 'metric':
            plt.plot(hist[key][:hist_size,0], '-',linewidth=0.8, label=key, color='C'+str(i))
    
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    plt.savefig(file, format='png')
    plt.close()
    