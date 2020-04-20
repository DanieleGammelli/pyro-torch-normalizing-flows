"""
Generates Data from a RotatingTwoMoons Dataset (i.e. applies specific rotation to the sklearn's TwoMoons Dataset)
"""
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class RotatingTwoMoonsConditionalSampler(object):
    def __init__(self, noise=0.05, random_seed=42):
        self.noise = noise
        #self.random_seed = random_seed
        self.rotation_sampler = torch.distributions.uniform.Uniform(low=0, high=2*np.pi)
        self.translation_sampler = torch.distributions.uniform.Uniform(low=-4, high=4)

    
    def conditioned_sample(self, n_samples=100,  theta=torch.tensor([0])):
        # Draw two moon samples and translate
        X, y = make_moons(n_samples=n_samples, shuffle=True, noise=self.noise)
        X_t = X - [0.5, 0.25]

        # Rotate points by theta radians
        X_r = self._rotate_points(X_t, theta).astype(np.float32)
        
        return  torch.from_numpy(X_r), torch.from_numpy(y)

    def conditioned_translated_sample(self, n_samples=100, theta=torch.tensor([0.]), trans=torch.tensor([0.,0.])):
        # Draw two moon samples, rotate and translate
        X, y = make_moons(n_samples=n_samples, shuffle=True, noise=self.noise)
        X_t = X - [0.5, 0.25]
        X_r = self._rotate_points(X_t, theta).astype(np.float32)
        
        X_r = torch.from_numpy(X_r)
        X_tf = X_r + trans
        
        return  X_tf.type(torch.FloatTensor), torch.from_numpy(y), theta, trans


    def joint_sample(self, n_samples):
        # Samples covariates
        theta = self.rotation_sampler.sample(sample_shape=[n_samples,1])
        
        # Draw two moon samples and translate
        X, y = make_moons(n_samples=n_samples, shuffle=True, noise=self.noise)
        X_t = X - [0.5, 0.25]
        
        # Rotate points based on covariates
        X_r = self._rotate_points(X_t, theta).astype(np.float32)
        
        return  torch.from_numpy(X_r), torch.from_numpy(y), theta

    def joint_translation_sample(self, n_samples):
        # Sample covariates
        theta = self.rotation_sampler.sample(sample_shape=[n_samples,1])
        trans = self.translation_sampler.sample(sample_shape=[n_samples, 2])

        # Draw two moon samples and rotate points
        X,y = make_moons(n_samples=n_samples, shuffle=True, noise=self.noise)
        X_t = X - [0.5, 0.25]
        X_r = self._rotate_points(X_t, theta).astype(np.float32)


        X_r = torch.from_numpy(X_r)
        X_tf = X_r + trans


        return X_tf.type(torch.FloatTensor), torch.from_numpy(y), theta, trans

    def make_plot(self, n_samples=100, theta=0):
        """
        Function used to make fancy plot
        """
        X, y = self.conditioned_sample(n_samples=n_samples, theta=theta, input_is_degrees=True)
        fig = plt.figure()
        axe = ax = fig.gca()
        axe.set_xlim(-2, 2)
        axe.set_ylim(-2, 2)
        sp, = axe.plot(X[:,0], X[:,1],color='k',marker='o',ls='')
        
        plt.show()
        return fig, axe, sp
    
    def update_plot(self, fig, axe, sp, n_samples=100, theta=0):
        """
        Function used to make fancy plot
        """
        X, y = self.conditioned_sample(n_samples=n_samples, theta=theta, input_is_degrees=True)
        sp.set_data(X[:,0],X[:,1])
        fig.canvas.draw()
                
    def _create_rotation_matrix(self, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        m = np.array([[c, -s], [s, c]])
        return m

    def _rotate_points(self, x_arr, theta):
        if theta.shape[0] == 1:
            m = self._create_rotation_matrix(theta)
            return np.array([np.dot(m,x) for x in x_arr])
        else:
            return np.array([np.dot(self._create_rotation_matrix(theta[i]), x_arr[i]) for i in range(len(x_arr))])
    
    def _degrees_to_radians(self, degrees):
        return degrees * np.pi / 180