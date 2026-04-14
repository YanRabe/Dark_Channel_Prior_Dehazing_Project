#dark channel computation
import os
import numpy as np
import cv2
import math

# Must be set before cv2 initializes
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import taichi as ti

# Initialize Taichi
ti.init(arch=ti.cpu)

# Normal Python function (no Taichi decorator) to handle OpenCV processing
def process_image(img_name: str):
    image = cv2.imread(img_name)
    if image is None:
        raise FileNotFoundError("L'image n'existe pas. Vérifiez le casing svp.")
        
    b, g, r = cv2.split(image)
    
    # Cast to int32 for easier min/max operations in Taichi
    r = r.astype(np.int32)
    g = g.astype(np.int32)
    b = b.astype(np.int32)

    n, m = b.shape
    
    # fenêtres 3x3
    dim = 3
    d = math.floor(dim / 2) 
    
    return n, m, d, b, g, r

# Taichi Kernel for parallel computation
@ti.kernel
def compute_dark_channel(
    n: ti.i32, 
    m: ti.i32, 
    d: ti.i32, 
    b: ti.types.ndarray(), 
    g: ti.types.ndarray(), 
    r: ti.types.ndarray(), 
    dark_channel: ti.types.ndarray()
):
    # This top-level loop over a 2D range is automatically parallelized by Taichi
    for i, j in ti.ndrange(n, m):
        local_min = 255  # Max possible value for an 8-bit image
        
        # Iterate over the spatial window (d x d)
        for di in range(-d, d + 1):
            for dj in range(-d, d + 1):
                # Clamp indices to the boundaries of the image so we don't go out of bounds
                ni = ti.max(0, ti.min(n - 1, i + di))
                nj = ti.max(0, ti.min(m - 1, j + dj))
                
                # Minimum among the 3 channels for this specific neighbor pixel
                p_min = ti.min(r[ni, nj], ti.min(g[ni, nj], b[ni, nj]))
                
                # Keep track of the minimum in the whole window
                local_min = ti.min(local_min, p_min)
                
        dark_channel[i, j] = local_min

# --- Main Execution ---
img_name = "bleak-désint.jpg"
n, m, d, b, g, r = process_image(img_name)

print("Image Dimensions:", n, m)

# Create the output numpy array
dark_channel_np = np.zeros((n, m), dtype=np.int32)

# Launch the Taichi kernel
compute_dark_channel(n, m, d, b, g, r, dark_channel_np)

# Convert the result back to uint8 for OpenCV rendering
dark_channel_out = dark_channel_np.astype(np.uint8)

cv2.imshow("dark channel", dark_channel_out)
cv2.imwrite("dark_channel.png", dark_channel_out)
cv2.waitKey(0)
cv2.destroyAllWindows()