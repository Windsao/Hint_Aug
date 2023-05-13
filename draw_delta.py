import numpy as np
import matplotlib.pyplot as plt

def normalize_vec(vec):
    return (vec - vec.min()) / (vec.max() - vec.min())

init_delta = np.load('init_delta.npy').squeeze()
final_delta = np.load('delta.npy').squeeze()
pre_delta = np.load('last_delta.npy').squeeze()

init_delta = normalize_vec(init_delta) 
final_delta = normalize_vec(final_delta) 
pre_delta = normalize_vec(pre_delta) 

plt.imshow(init_delta.transpose(1,2,0))
plt.savefig('init')
plt.close()
plt.imshow(final_delta.transpose(1,2,0))
plt.savefig('final')
plt.close()
plt.imshow(pre_delta.transpose(1,2,0))
plt.savefig('pre')
plt.close()