import numpy as np
import argparse
import cv2

def difX(mat):
    return np.concatenate((mat[:,1:], mat[:,-1:]), axis=1) - mat

def difY(mat):
    return np.concatenate((mat[1:,:], mat[-1:,:]), axis=0) - mat

def divX(mat):
    R = mat - np.concatenate((mat[:,:1], mat[:,:-1]), axis=1)
    R[:,0] = mat[:,0] 
    R[:,-1] = -mat[:,-2]
    return R

def divY(mat):
    R = mat - np.concatenate((mat[:1,:], mat[:-1,:]), axis=0)
    R[0,:] = mat[0,:] 
    R[-1,:] = -mat[-2,:]
    return R

def boxproj(mat):
    return np.clip(mat, 0, 1)

def threasholding_l1_w(X, Y):
    return -np.sign(X) * np.clip(0, None, np.abs(X)-Y)

def theasholding_RS(X, Y, a, b):
    normRS = np.sqrt(X ** 2 + Y ** 2)
    W = np.clip(0, None, 1 - (a / b) / normRS)
    R = -X * W
    S = -Y * W
    return R, S

def main():
    parser = argparse.ArgumentParser(description='L0TV:')
    parser.add_argument('--image', '-i', type=str, help='Input image', required=True)
    parser.add_argument('--noise', '-n', type=int, default=5, help='noise level')
    args = parser.parse_args()

    src = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if src is None:
        print('File is not found.')
        return

    n = np.random.randint(0, 100, size=src.shape)
    src[n < args.noise] = 0
    src[n > 100 - args.noise] = 255

    B = src.astype(np.float) / 255
    U = B.copy()
    dx = difX(U)
    dy = difY(U)
    Kub = U - B
    V = np.ones_like(B)
    Z = Kub.copy()
    R = difX(U)
    S = difY(U)

    piz = np.zeros_like(B)
    pir = np.zeros_like(B)
    pis = np.zeros_like(B)
    piv = np.zeros_like(B)

    gamma = 0.5 * (1 + np.sqrt(5.0))
    alpha = 10.0
    beta = 10.0
    rho = 10.0
    ratio = 3.0

    lm = 8.0
    LargestEig = 1.0
    max_itr = 100
    
    for itr in range(max_itr):
        cof_A = rho * V ** 2 + alpha
        cof_B = -alpha * Kub - piz
        cof_C = piv * V
        Z = threasholding_l1_w(cof_B / cof_A, cof_C / cof_A)

        # UpdateRS
        R, S = theasholding_RS(-pir / beta - dx, -pis / beta - dy, lm, beta)

        # Update U
        g1 = alpha * (Kub-Z) + piz
        g3 = -beta*divX(dx) + divX(-pir + beta*R)
        g4 = -beta*divY(dy) + divY(-pis + beta*S)
        gradU = g1 + g3 + g4
        Lip = beta*4 + beta*4 + alpha*LargestEig
        U = boxproj(U - gradU/Lip)
        dx = difX(U)
        dy = difY(U)
        Kub = U - B

        # Update V
        cof_A = rho * Z ** 2 + 1.0e-10
        cof_B = piv * np.abs(Z) - 1
        cof_C = -cof_B / cof_A
        V = boxproj(cof_C)

        Kubz = Kub - Z
        dxR = dx - R
        dyS = dy - S
        VabsZ = V * np.abs(Z)

        piz = piz + gamma * alpha * Kubz
        pir = pir + gamma * beta * dxR
        pis = pis + gamma * beta * dyS
        piv = piv + gamma * rho * VabsZ

        r1 = np.linalg.norm(Kubz)
        r2 = np.linalg.norm(dxR) + np.linalg.norm(dyS)
        r3 = np.linalg.norm(VabsZ)

        if (itr+1) % 30 == 0:
            if r1 > r2 and r1 > r3: 
                alpha = alpha * ratio
            if r2 > r1 and r2 > r3: 
                beta = beta * ratio
            if r3 > r1 and r3 > r2: 
                rho = rho * ratio

        cv2.imshow('image', src)
        cv2.imshow('denoise', U)
        cv2.waitKey(1)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
