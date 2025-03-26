import numpy as np

class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.3, dt=1e-2, x0=None):
        """
        Parameters:
            size (int or tuple): The shape of the noise vector.
            mu (float or array): The long-term mean. Typically set to 0.
            theta (float): Rate of mean reversion.
            sigma (float): Volatility parameter.
            dt (float): Time step for discretization.
            x0 (float or array): Initial value of the noise process. If None, initialized to zeros.
        """
        self.size = size
        self.mu = np.full(shape=size, fill_value=mu) if isinstance(mu, (int, float)) else np.array(mu)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0 if x0 is not None else np.zeros(size)
        self.reset()

    def reset(self):
        """Reset the internal state to the initial value (or zeros)."""
        self.x_prev = np.copy(self.x0)
    
    def __call__(self):
        """Generate the next noise value."""
        # Generate noise sampled from a normal distribution
        noise = np.random.normal(size=self.size)
        # Compute the new value based on the OU update
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * noise
        self.x_prev = x
        return x

class Gaussian_noise:
    def __init__(self, shape, mean = 0.0, std = 0.1):
        self.mean = mean
        self.std = std
        self.shape = shape

    def __call__(self):
            return np.random.normal(loc=self.mean, scale=self.std, size=self.shape)