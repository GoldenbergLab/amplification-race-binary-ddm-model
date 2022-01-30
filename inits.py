def init_f(modelcode, nsubs):
    res = 0
    if modelcode == 'ddmcode':
        res = {'mu_pr': [1, 1, 0, 0, 0, 0, -1],
            'sigma': [1]*7,
            'alpha_pr': [[1]*nsubs]*2,
            'zed_pr': [[0]*nsubs]*2,
            'delta_pr': [[0]*nsubs]*2,
            'tau_pr': [-1]*nsubs
        }
    elif modelcode == 'ddmracecode':
        res = {'mu_pr': [.5, .5, 0, 0, 0, 0, 0, -1],
            'sigma': [1]*8,
            'alpha_pr': [[.5]*nsubs]*2,
            'zed_pr': [[0]*nsubs]*2,
            'delta_pr': [[0]*nsubs]*3,
            'tau_pr': [-1]*nsubs
        }
    elif modelcode == 'ddmraceallcode':
        res = {'mu_pr': [.5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1], #ddmraceall
            'sigma': [1]*19,
            'alpha_pr': [[.5]*nsubs]*6,
            'zed_pr': [[0]*nsubs]*6,
            'delta_pr': [[0]*nsubs]*6,
            'tau_pr': [-1]*nsubs
        }
    elif modelcode == 'facesddmcode':
        res = {'mu_pr': [.5, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            'sigma': [1]*15,
            'alpha_pr': [[.5]*nsubs]*2,
            'zed_pr': [[0]*nsubs]*2,
            'lambda_pr': [[0]*nsubs]*2,
            'beta_pr':[[0]*nsubs]*4,
            'gamma_pr':[[0]*nsubs]*4,
            'tau_pr': [-1]*nsubs
        }
    return res
