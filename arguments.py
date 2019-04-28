import argparse
import torch


def argparser():
    parser = argparse.ArgumentParser(description='Ape-X')

    # Common Arguments
    parser.add_argument('--seed', type=int, default=1122,
                        help='Random seed')
    parser.add_argument('--n_steps', type=int, default=3,
                        help='Number of steps in multi-step learning')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for multi-step learning')

    # Environment Arguments
    parser.add_argument('--env', type=str, default="SeaquestNoFrameskip-v4",
                        help='Atari environment to use')
    parser.add_argument('--episode_life', type=int, default=1,
                        help='Whether env has episode life(1) or not(0)')
    parser.add_argument('--clip_rewards', type=int, default=1,
                        help='Whether env clip rewards(1) or not(0)')
    parser.add_argument('--frame_stack', type=int, default=1,
                        help='Whether env stacks frame(1) or not(0)')
    parser.add_argument('--scale', type=int, default=0,
                        help='Whether env scales(1) or not(0)')

    # Arguments for Actor
    parser.add_argument('--send_interval', type=int, default=50,
                        help='Number of samples batch to be transferred to replay will contain')
    parser.add_argument('--update_interval', type=int, default=400,
                        help='Interval of fetching parameters from learner')
    parser.add_argument('--max_episode_length', type=int, default=50000,
                        help='Maximum length of episode')
    parser.add_argument('--max_outstanding', type=int, default=3,
                        help='Maximum number of outstanding batch push requests')
    parser.add_argument('--eps_base', type=float, default=0.4)
    parser.add_argument('--eps_alpha', type=float, default=7.0)

    # Arguments for Replay
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Priority exponent')
    parser.add_argument('--beta', type=float, default=0.4,
                        help='Importance sampling exponent')
    parser.add_argument('--replay_buffer_size', type=int, default=2000000,
                        help='Size of prioritized replay buffer')
    parser.add_argument('--threshold_size', type=int, default=50000,
                        help='Threshold for starting to transfer batches to learner')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Size of samples prefetched batches will contain')
    parser.add_argument('--n_recv_batch_worker', type=int, default=4)
    parser.add_argument('--n_recv_prios_worker', type=int, default=4)
    parser.add_argument('--n_send_batch_worker', type=int, default=8)

    # Arguments for Learner
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--queue_size', type=int, default=16,
                        help='Size of local queue. If this value is set to 16, '
                             'local queue can contain up to 16 prefetched batches')
    parser.add_argument('--prios_queue_size', type=int, default=16)
    parser.add_argument('--max_norm', type=float, default=40.0,
                        help='Maximum gradient norm to clip')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--target_update_interval', type=int, default=2500,
                        help='Interval of updating target network')
    parser.add_argument('--publish_param_interval', type=int, default=25,
                        help='Interval of publishing parameter to actors')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='Interval of saving model parameters')
    parser.add_argument('--bps_interval', type=int, default=100,
                        help='Interval of logging BPS')
    parser.add_argument('--n_recv_batch_process', type=int, default=4,
                        help='Number of processes to receive batch from replay')

    # Arguments for Evaluation
    parser.add_argument('--render', action='store_true', default=False)

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args
