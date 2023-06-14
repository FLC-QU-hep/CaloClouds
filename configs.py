class Configs():
    
    def __init__(self):
        
    # Comet Experiment Name
        self.log_comet = False
        self.name = 'CaloClouds'

    # Model arguments
        self.latent_dim = 256
        self.num_steps = 100
        self.beta_1 = 1e-4
        self.beta_T = 0.02
        self.sched_mode = 'quardatic'  # options: ['linear', 'quardatic', 'sigmoid]
        self.flexibility = 0.0
        self.truncate_std = 2.0
        self.latent_flow_depth = 14
        self.latent_flow_hidden_dim = 256
        self.num_samples = 4
        self.features = 4
        self.sample_num_points = 2048
        self.kl_weight = 0.001
        self.residual = True  
        
        self.cond_features = 2       # number of conditioning features (i.e. energy+points=2)
        self.norm_cond = True    # normalize conditioniong to [-1,1]
        self.kld_min = 1.0 

        # EPiC arguments
        self.use_epic = False
        self.epic_layers = 5
        self.hid_d = 128
        self.sum_scale = 1e-3
        self.weight_norm = True

        # for n_flows model
        self.flow_model = 'PiecewiseRationalQuadraticCouplingTransform'
        self.flow_transforms = 10
        self.flow_layers = 2
        self.flow_hidden_dims = 128
        self.tails = 'linear'
        self.tail_bound = 10
    

    # Data
        self.dataset = 'x36_grid' # choices=['x36_grid', 'clustered']
        self.dataset_path = './data/calo-clouds/hdf5/high_granular_grid/train/10-90GeV_x36_grid_regular_524k_float32.hdf5'
        self.quantized_pos = False

    # Dtataloader
        self.workers = 32
        self.train_bs = 128
        self.pin_memory = False 
        self.shuffle = True
        self.max_points = 6_000

    # Optimizer and scheduler
        self.lr = 2e-3
        self.weight_decay = 0
        self.max_grad_norm = 10
        self.end_lr = 1e-4
        self.sched_start_epoch = 300 * 1e3
        self.sched_end_epoch = 2 * 1e6

    # Others
        self.device = 'cuda'
        self.logdir = './log'
        self.seed = 42
        self.max_iters = 2 * 1e6
        self.val_freq = 1e3
        self.test_freq = 30 * 1e3
        self.test_size = 400
        self.tag = None
        

    