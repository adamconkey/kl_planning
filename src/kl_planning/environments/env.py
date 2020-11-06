
class Environment:
    """
    Base class for simulation environments.
    """

    def __init__(self, config, m_projection=False, belief_dynamics_noise=0.02,
                 device=torch.device('cuda')):
        """
        Args:
            config (dict): Configuration for environment (loaded from YAML)
            m_projection (bool): Use M-projection for KL divergence if True, otherwise I-projection
            belief_dynamics_noise (float): Noise gain factor on belief dynamics
            device (torch.device): Device to perform torch computations on
        """
        self.object_config = config['objects']
        self.agent_config = config['agent']
        self.indicator_config = config['indicators']
        self.start_config = config['start']
        self.goal_config = config['goals']
        self.state_size = len(self.start_config['state'])

        self.m_projection = m_projection
        self.belief_dynamics_noise = belief_dynamics_noise
        self.device = device
        
    def get_start_state(self):
        """
        Returns start state from environment configuration.

        Returns:
            state (list): Start state (x,y,theta)
        """
        return self.start_config['state']

    def get_start_covariance(self):
        """
        Returns start state covariance from environment configuration.

        Returns:
            covariance (list): Diagonal covariance values for state (x,y,theta)
        """
        return self.start_config['covariance']

    def get_goal_states(self):
        """
        Returns list of goal states from environment configuration.

        Returns:
            goal_states (list): List of goal states (each a list (x,y,theta))
        """
        return [v['state'] for v in self.goal_config.values()]

    def get_goal_covariances(self):
        """
        Returns list of goal state covariances from environment configuration.
        
        Returns:
            goal_covariances (list): List of diagonal covariances for goal states
        """
        return [v['covariance'] for v in self.goal_config.values()]

    def get_goal_weights(self):
        """
        Returns component weights for GMM goal distribution.

        Returns:
            goal_weights (list): List of component weights for GMM goal distribution
        """
        return [v['weight'] for v in self.goal_config.values()]

    def get_goal_low_high(self):
        """
        Returns low/high values for uniform goal distribution.

        Returns:
            low_high (list): List of low/high values for uniform goal distribution
        """
        return [self.goal_config['goal']['low'], self.goal_config['goal']['high']]
        
    def set_agent_location(self, state):
        """
        Sets the agents state location in the simulation environment.
        
        Args:
            state (list): State to set the current agent's location to
        """
        raise NotImplementedError()
        
    def dynamics(self, state, act, noise_gain):
        """
        Nonlinear stochastic dynamics function for the agent.

        Args:
            state (Tensor): Start state of shape (n_batch, n_state)
            act (Tensor): Action to apply of shape (n_batch, n_act)
            noise_gain (float): Gain factor for additive Gaussian noise to dynamics
        Returns:
            next_state (Tensor): Next state after applying action on current state and feeding
                                 through nonlinear stochastic dynamics of shape (n_batch, n_state)
        """
        raise NotImplementedError()
        
    def cost(self, act, start_dist, goal_dist, kl_divergence):
        """
        Cost function used by planner to rank CEM samples. Must be implemented by each environment.

        Args:
            act (Tensor): Actions to be applied of shape (horizon, n_candidates, n_act)
            start_dist (distribution): Start state distribution
            goal_dist (distribution): Goal state distribution
            kl_divergence (function): KL divergence function defined for state/goal distributions.
        Returns:
            cost (Tensor): Computed costs of shape (n_candidates,)
        """
        raise NotImplementedError()
