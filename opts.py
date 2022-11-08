import argparse
def get_args():
    parser = argparse.ArgumentParser(description='QA')

    """Optimization related arguments"""
    optim_args = parser.add_argument_group('Optimization related arguments')
    optim_args.add_argument('--train_batch_size', type=int,  default= 32, help='Training batch Size')
    optim_args.add_argument('--eval_batch_size', type=int,  default= 1, help='Evaluation batch Size')
    optim_args.add_argument('--top_k', type=float,  default= 0., help='top k for infer')
    optim_args.add_argument('--top_p', type=float,  default= 0.9, help='top p for infer')
    optim_args.add_argument('--weight_decay', type=float,  default= 0., help='Weight decay')
    optim_args.add_argument('--num_train_epochs', type=int,  default= 30, help='Training epoch')
    optim_args.add_argument('--learning_rate', type=float,  default= 2e-5, help='Learning rate')
    optim_args.add_argument('--gradient_accumulation_steps', type=int,  default= 1, help='Gradient accumulation')
    optim_args.add_argument('--start_epoch', type=int,  default= 0, help='start epoch')
    optim_args.add_argument('--warmup_ratio', type=float,  default= 0.6, help='Learning rate')
    optim_args.add_argument('--val_check_interval', type=float,  default= 0.5, help='Validation check interval')

    
    """Data related arguments"""
    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument('--img_size', type=int, default=224, help='Image size')
    data_args.add_argument('--ckpt_path', type=str, default='/media/storage/checkpoints/NLX_GPT', help='checkpoint path')
    data_args.add_argument('--cached_dir', type=str, default='cache', help='dataset cache path')
    data_args.add_argument('--caption_save_path', type=str, default='cococaption/results/', help='Caption result save path')
    data_args.add_argument('--annFileExp', type=str, default='cococaption/annotations/vqaX_test_annot_exp.json', help='Image size')
    data_args.add_argument('--annFileFull', type=str, default='cococaption/annotations/vqaX_test_annot_full.json', help='Image size')
    data_args.add_argument('--data_path', type=str, default='', help='vqax train dataset path')
    data_args.add_argument('--nle_image_dir', type=str, default='', help='Image dataset path')

    
    """Model related arguments"""
    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument('--finetune_pretrained', action="store_true", help='if True, finetunes from the image captioning model')
    model_args.add_argument('--load_from_epoch', type=int, default=None, help='load checkpoint epoch number')
    model_args.add_argument('--no_sample', action="store_true", help='No sampling')
    model_args.add_argument('--max_seq_len', type=int, default=40, help='Max sequence length')
    model_args.add_argument('--relevance_map', action="store_true", help='Visualizing relevance map')
    model_args.add_argument('--adapter_latent_size', type=int, default=None, help='Adapter latent size')

    """Logging related arguments"""
    misc_args = parser.add_argument_group('Logging related & Misc arguments')
    misc_args.add_argument('--seed', type=int, default=42, help='Random Seed')
    misc_args.add_argument('--project_name', type=str, default='NLX-GPT', help='Project name for wandb')
    misc_args.add_argument('--experiment_name', type=str, default='NLX-GPT', help='Experiment name for wandb')
    misc_args.add_argument('--ngpu', type=int, default=1, help='Number of gpu')
    misc_args.add_argument('--mode', type=str, default=None, help='Train or Test')
    misc_args.add_argument('--boost_mode', type=str, default="apex", help='Accelator')
    misc_args.add_argument('--local_rank', type=int, default=0, help='Local rank')
    misc_args.add_argument('--distributed', action="store_true", help='Distributed devices')
    misc_args.add_argument('--opt_level', type=str, default="O1", help='Accelator')
    misc_args.add_argument('--train_mode', type=str, default=None, help='Prompt modification')


    args = parser.parse_args()
    return args
