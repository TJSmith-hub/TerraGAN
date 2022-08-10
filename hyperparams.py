UNET_GAN_PARAMS = {
    # dataset and data loader
    'train_path': 'dataset/data_train',
    'img_size': 256,
    'batch_size': 25,
    'n_workers': 6,
    'pin_memory': True,
    
    # training
    'display_every': 1,
    'save_fig_every': 1,
    'epochs': 100,
    'lr_G': 0.0002,
    'lr_D': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'lambda_L1': 100,
    
    # generator
    'input_c_G': 3,
    'output_c_G': 4,
    'n_down_G': 6,
    'n_filters_G': 64,
    
    # discriminator
    'input_c_D': 7,
    'n_down_D': 4,
    'n_filters_D': 64,
}