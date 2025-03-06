import sys
print("Python version:", sys.version)

try:
    import skimage
    print("scikit-image version:", skimage.__version__)
    print("Available modules in skimage:", dir(skimage))
    
    try:
        from skimage import metrics
        print("Available functions in metrics:", dir(metrics))
        
        try:
            from skimage.metrics import structural_similarity as ssim
            from skimage.metrics import peak_signal_noise_ratio as psnr
            print("Successfully imported SSIM and PSNR")
        except ImportError as e:
            print("Error importing SSIM/PSNR:", e)
    except ImportError as e:
        print("Error importing metrics module:", e)
except ImportError as e:
    print("Error importing skimage:", e)

print("Test completed") 