class hparams:

    class run:
        name = "binarized_mnist_baseline"
        # run.seed: seed that fixes all randomness in the project
        seed = 420

        # Hardware
        # Global run config for GPUs and CPUs
        num_gpus = 1
        num_cpus = 256

        # JAX only: Defines how many checkpoints will be kept on disk (the latest N)
        max_allowed_checkpoints = 5

    class data:
        dataset_source = 'binarized_mnist'

        # Data paths. Not used for (binarized_mnist, cifar-10)
        train_data_path = '../datasets/imagenet_32/train_data/'
        val_data_path = '../datasets/imagenet_32/val_data/'
        synthesis_data_path = '../datasets/imagenet_32/val_data/'

        # Image metadata
        # Image resolution of the dataset (High and Width, assumed square)
        target_res = 32
        # Image channels of the dataset (Number of color channels)
        channels = 1
        # Image color depth in the dataset (bit-depth of each color channel)
        num_bits = 8.
        # Whether to do a random horizontal flip of images when loading the data (no applicable to MNIST)
        random_horizontal_flip = True
    
    class model:
        # Model section: Defines the model design (architecture) hyper-parameters
        # Some of these parameters will likely never be changed.

        # General
        # Main experimentation params
        # Whether to apply a scaling by 1/sqrt(L) at the end of each residual bottleneck block (minimal effect on stability)
        stable_init = True
        # Whether or not to intialize the prior latent layer as zeros (no effect)
        initialize_prior_weights_as_zero = False
        # Whether to use 1x1 convs in the beginning and end of the residual bottleneck block (effective at reducing the memory load)
        use_1x1_conv = True

        # Latent layer distribution base can be in ('std', 'logstd'). Determines if the model should predict std (with softplus) or logstd (std is computed with exp(logstd)).
        distribution_base = 'std'
        # Similarly for output layer
        output_distribution_base = 'std'
        # Latent layer Gradient smoothing beta. ln(2) ~= 0.6931472. Setting this parameter to 1. disables gradient smoothing (not recommended)
        gradient_smoothing_beta = 0.6931472
        # Similarly for output layer
        output_gradient_smoothing_beta = 0.6931472

        ################################# Layers' structure parameters ####################################
        # In the bottom-up block, a skip connection is only taken once every n_blocks_per_res + 1 residual blocks
        # That skip connection is linked to all top-down blocks of the matching resolution defined by backwards index in down_strides
        # Example: up_strides = [x, a, x, x] and down_strides = [y, y, b, y] | the skip from a is linked to all blocks of b.

        # Both downsampling and upsampling is done in tandem with residual blocks in a downsample-last upsample-first manner.
        # i.e, both the downsample block and the upsample block are considered to belong to the higher resolution.
        # Example: up_strides = [2, 2, 4] and up_n_blocks_per_res = [5, 9, 3] means there are in reality [6, 10, 4] blocks in their respective resolutions
        # Similarly for down_strides = [4, 2, 2] and down_n_blocks_per_res = [3, 9, 5] means there are in reality [6, 10, 4] blocks.

        # The residual blocks in up_n_blocks_per_res are always run BEFORE the residual block in up_strides.
        # Inversely, the residual blocks in down_n_blocks_per_res are always run AFTER the residual block in down_strides.
        # Downsampling of images happens in the residual block of up_strides. Upsampling of images happens in the residual block of down_strides.
        # Skip connections between bottom-up and top-down blocks are taken once every time an up_strides block is called.

        # down_strides must always be equal to the inverted up_strides. but any symmetry beyond that is not necessary (Appendix C.2 of the paper)
        # To design a model that only links the bottom-up and top-down blocks once per resolution (like VDVAE), one should define only one up_stride layer
        # per resolution and define as many up_n_blocks_per_res before each as they want. down_strides is the symmetrical counterpart to up_strides and down_n_blocks_per_res
        # should contain as many block that come after an up_stride as one desires.
        # Example: up_strides = [2, 2, 4, 1], up_n_blocks_per_res = [4, 5, 9, 3], down_strides = [1, 4, 2, 2], down_n_blocks_per_res = [4, 4, 11, 7] and a target_res=32
        # means that bottom-up and top-down are linked once in resolution 32x32, once in 16x16, once in 8x8 and once in 2x2. Bottom-up and top-down networks however don't have
        # the same number of layers.
        # To design a model that links every bottom-up block to every top-down block (forces architectural symmetry), one should ensure that up_n_blocks_per_res and
        # down_n_blocks_per_res are all zeros.
        # Example: up_strides = [1, 1, 2, 1, 2, 1, 1, 4, 1], down_strides = [1, 4, 1, 1, 2, 1, 2, 1, 1], up_n_blocks_per_res = down_n_blocks_per_res = [0] * 9 and a target_res=32
        # means that we design 3 layers in resolution 32x32, 2 layers in 16x16, 3 layers in 8x8 and 1 layer in 2x2. All these layers are connected between the bottom-up and top-down blocks.
        # It's also possible to design any combination of the two above examples (partial symmetry).
        # Example: up_strides = [2, 2, 1, 1, 4, 1], up_n_blocks_per_res = [4, 5, 0, 0, 0, 0], down_strides = [1, 4, 1, 1, 2, 2], down_n_blocks_per_res = [0, 0, 2, 3, 7, 4] and a target_res=32

        # Bottom-up block
        # Input conv
        input_conv_filters = 32
        input_kernel_size = (1, 1)

        # Bottom-up blocks
        # ALL LISTS MUST HAVE THE SAME LENGTH
        # Defines the stride value of each block that will send a connection to top-down
        up_strides = [2] + [1, 2] + [1] * 9 + [2] + [1] * 4 + [4] + [1] * 5
        # Defines the number of residual blocks prior to a connection with top-down
        up_n_blocks_per_res = [0] * 23
        # Defines the number of residual bottleneck blocks per bottom-up block (usually kept at 1)
        up_n_blocks = [1] * 23
        # Defines the number of middle layers in the residual bottleneck block (doesn't account for the first and last layer of the res block)
        up_n_layers = [2] * 23
        # Defines the filter size of the model per bottom-up block. If up_stride != 1, then the filter size can be changed from that of the input, else it is ignored.
        up_filters = [64, 64] + [128] * 9 + [128] + [256] * 4 + [256] + [512] * 6
        # Defines the bottleneck ratio inside the bottleneck res block
        up_mid_filters_ratio = [1.] * 23
        # Defines the kernel size of the convs of the res block
        up_kernel_size = [3] * 18 + [1] * 5
        # Defines the filter size of the skip projection (projection of the activation sent to top-down).
        up_skip_filters = [32] + [64, 64] + [128] * 9 + [128] + [256] * 4 + [256] + [512] * 5

        # Latent layers
        # Whether to use the residual distribution from NVAE (no effect on stability, NLL or memory)
        use_residual_distribution = False

        # Top-down blocks
        # ALL LISTS MUST HAVE THE SAME LENGTH
        # Defines the strides value of each top-down block that will be the first to receive an activation from bottom-up
        # must be symmetric (inverted) form of up_strides
        down_strides = [1] * 5 + [4] + [1] * 4 + [2] + [1] * 9 + [2, 1] + [2]
        # Defines the number of residual blocks after a first connection with bottom-up (number of res blocks that re-use the same bottom-up activation)
        down_n_blocks_per_res = [0] * 23
        # Defines the number of residual bottleneck blocks per top-down block (usually kept at 1)
        down_n_blocks = [1] * 23
        # Defines the number of middle layers in the residual bottleneck block (doesn't account for the first and last layer of the res block)
        down_n_layers = [2] * 23
        # Defines the filter size of the model per top-down block. If down_strides != 1, then the filter size can be changed from that of the input, else it is ignored.
        down_filters = [512] * 5 + [256] + [256] * 4 + [128] + [128] * 9 + [64, 64] + [32]
        # Defines the bottleneck ratio inside the bottleneck res block
        down_mid_filters_ratio = [1.] * 23
        # Defines the kernel size of the convs of the res block
        down_kernel_size = [1] * 5 + [3] * 18
        # Defines the number of gaussian variates per top-down block
        down_latent_variates = [32] * 23

        # Output conv
        output_kernel_size = (1, 1)
        num_output_mixtures = 10