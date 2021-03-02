from text import symbols


class hparams:
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500
        iters_per_checkpoint=5000
        seed=4321
        synth_batch_size = 1

        ################################
        # Data Parameters             #
        ################################
        mel_training_files='./training_data/mel-bznsyp_character_pinyin_data_train.txt'
        mel_validation_files='./training_data/mel-bznsyp_character_pinyin_data_val.txt'

        ################################
        # Audio Parameters             #
        ################################
        
        # null

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols)

        # Encoder parameters
        encoder_embedding_dim=512
        encoder_kernel_size=5
        encoder_n_convolutions=3
        encoder_input_dim = [512, 512, 512]
        encoder_output_dim=[512, 512, 512]

        # Decoder parameters
        n_frames_per_step=3
        decoder_rnn_dim=1024
        prenet_dim=[256, 256]
        max_decoder_steps=1000
        gate_threshold=0.5
        p_attention_dropout=0.1
        p_decoder_dropout=0.1

        # Attention parameters
        attention_rnn_dim=1024
        attention_dim=128

        # Location Layer parameters
        location_n_filters=32
        location_kernel_size=31

        # Mel-post processing network parameters
        postnet_embedding_dim=512
        postnet_kernel_size=5
        postnet_n_convolutions=5

        ################################
        # Optimization Hyperparameters #
        ################################
        learning_rate=1e-3
        weight_decay=1e-6
        grad_clip_thresh=1.0
        batch_size=32


        ################################
        # training setting             #
        ################################
        output_directory = './ckpt_model'
        log_directory = './log_dir'
        checkpoint_path = None
