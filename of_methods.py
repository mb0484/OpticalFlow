from ex1_utils import gaussderiv, gausssmooth
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2

def lucas_kanade(im1, im2, N, sigma=0.45, threshold=1e-7):
    
    im1 = im1 / 255.
    im2 = im2 / 255.
    
    Ix, Iy = gaussderiv(np.divide(
            np.add(
                im1,
                im2
            ),
            2.0
        ),
        sigma
    )
    
    It = gausssmooth(np.subtract(im2, im1), 1.0)
    
    Ix_t = np.multiply(Ix, It)
    Iy_t = np.multiply(Iy, It)
    Ix_2 = np.square(Ix)
    Iy_2 = np.square(Iy)
    Ix_y = np.multiply(Ix, Iy)
    
    sum_kernel = np.ones((N, N))
    
    sum_Ix_y = cv2.filter2D(Ix_y, -1, sum_kernel)
    sum_Ix_t = cv2.filter2D(Ix_t, -1, sum_kernel)
    sum_Iy_t = cv2.filter2D(Iy_t, -1, sum_kernel)
    sum_Ix_2 = cv2.filter2D(Ix_2, -1, sum_kernel)
    sum_Iy_2 = cv2.filter2D(Iy_2, -1, sum_kernel)
    sum_Ix_y_2 = np.square(sum_Ix_y)
    
    D = np.subtract(
            np.multiply(
                sum_Ix_2,
                sum_Iy_2
            ),
            sum_Ix_y_2
        )
    
    D[D <= threshold] = threshold
    
    u = np.divide(
        np.add(
            np.multiply(
                - sum_Iy_2,
                sum_Ix_t
            ),
            np.multiply(
                sum_Ix_y,
                sum_Iy_t
            )
        ),
        D
    )
    
    v = np.divide(
        np.subtract(
            np.multiply(
                sum_Ix_y,
                sum_Ix_t
            ),
            np.multiply(
                sum_Ix_2,
                sum_Iy_t
            )
        ),
        D
    )
    
    return u, v


def horn_schunck(im1, im2, n_iters, lmdb, N = None):
    #normalize
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    
    Ix, Iy = gaussderiv(
        np.divide(
            np.add(
                im1,
                im2
            ),
            2
        ),
        0.4
    )
    
    It = gausssmooth(np.subtract(im2, im1), 1.0)
    
    Ix_2 = np.square(Ix)
    Iy_2 = np.square(Iy)
    
    L_d = np.matrix([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    
    if N:
        u, v = lucas_kanade(im1, im2, N)
    else:
        u = np.zeros(im1.shape);
        v = np.zeros(im2.shape);
    
    D = np.add(
        np.add(
            lmdb,
            Ix_2
        ),
        Iy_2
    )
    
    D[D <= 1e-7] = 1e-7
    
    start = True
    
    for iter in range(n_iters):
        if iter % 100 == 0:
            
            if not start:
                u_sim = np.sum(cosine_similarity(u, u_old) / (Ix.shape[0] * Iy.shape[1]))
                v_sim = np.sum(cosine_similarity(v, v_old) / (Ix.shape[0] * Iy.shape[1]))
                u_old = u
                v_old = v
                
                if u_sim > 0.90 or v_sim > 0.90:
                    break
            else:
                u_old = u.copy()
                v_old = v.copy()
                start = False
        
        u_a = cv2.filter2D(u, -1, L_d)
        v_a = cv2.filter2D(v, -1, L_d)
        
        P = np.add(
            np.add(
                np.multiply(
                    Ix,
                    u_a
                ),
                np.multiply(
                    Iy,
                    v_a
                )
            ),
            It
        )
        
        P_D = np.divide(
            P,
            D
        )
        
        u = np.subtract(
            u_a,
            np.multiply(
                Ix,
                P_D
            )
        )
        
        v = np.subtract(
            v_a,
            np.multiply(
                Iy,
                P_D
            )
        )
    
    
    return u, v