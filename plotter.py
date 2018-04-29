import matplotlib.pyplot as plt
import pdb

def plot(beta_val, alphas, psnrs, ssims, save = True):
    fname= "beta_{b_val}.png".format(b_val= str(beta_val))
    psnr_fig = plt.figure()
    plt.subplot(2,1,1)
    plt.title("beta = {b_val}".format(b_val = beta_val))
    plt.ylabel('psnr')
    plt.plot(alphas, psnrs,'o-')
    plt.subplot(2,1,2)
    plt.xlabel('alpha')
    plt.ylabel('ssim')
    plt.plot(alphas, ssims,'o-')
    plt.savefig(fname)
    plt.cla()
    plt.close()

beta_val = 1
alphas = [1,2,3,4,5]
psnrs = [1,2,3,4,5]
ssims = [10,20,30,40,50]
plot(1,[1,2,3,4,5] ,[1,2,3,4,5],[10,20,30,40,50])