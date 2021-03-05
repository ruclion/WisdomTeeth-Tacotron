class hparams:

        # Audio Parameters             
        n_mel_channels = 80


        # Phoneme lookup-table parameters
        n_symbols=35


        # Encoder parameters
        encoder_embedding_dim=512
        encoder_n_convolutions=3
        encoder_input_dim = [512, 512, 512]
        encoder_output_dim=[512, 512, 512]

        encoder_kernel_size=5
        encoder_cnn_dropout_p=0.5


        # Attention-Decoder parameters
        n_frames_per_step=1

        attention_rnn_dim=1024
        attention_rnn_dropout_p=0.1

        decoder_rnn_dim=1024
        decoder_rnn_dropout_p=0.1

        prenet_dim=[256, 256]
        prenet_dropout_p = 0.5

        gate_threshold=0.5

        attention_dim=128
        location_n_filters=32
        location_kernel_size=31


        # Mel-post processing network parameters
        postnet_num_convolutions=5
        postnet_embedding_dim_in=[80, 512, 512, 512]
        postnet_embedding_dim_out=[512, 512, 512, 512]
        final_postnet_embedding_dim_in=512
        final_postnet_embedding_dim_out=80

        postnet_kernel_size=5
        postnet_dropout_p=0.5
        


        # Hyperparameters 
        batch_size=48
        learning_rate=1e-3
        weight_decay=1e-6
        grad_clip_thresh=1.0


        output_directory = './ckpt_model'
        log_directory = './log_dir'
        inference_directory = './infer_eval_dir'
        checkpoint_path = None


        epochs=1
        seed=4321 
        mel_training_files='./preprocess_dataset/training_data/train.txt'
        mel_validation_files='./preprocess_dataset/training_data/val.txt'
        mel_test_files='./preprocess_dataset/training_data/test.txt'