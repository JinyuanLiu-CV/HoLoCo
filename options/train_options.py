from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--mse_weight', type=float, default=1)
        self.parser.add_argument('--vgg_weight',type=float, default=0.1)
        self.parser.add_argument('--gan_weight',type=float,default=0.2)
        self.parser.add_argument('--contract_weight',type=float,default=0.2)
        self.parser.add_argument('--global_local_rate',type=float,default=4)
        self.parser.add_argument('--display_freq', type=int, default=30, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', type=int, default=0,help='continue training: load the latest model')
        self.parser.add_argument('--continue_epoch',type=int,default=0,help='which epoch do you want to restore')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--config', type=str, default='configs/unit_gta2city_folder.yaml', help='Path to the config file.')
        self.parser.add_argument('--fullinput',type=int, default=0,help='3+ photos contract loss')
        self.parser.add_argument('--ssim_loss',type=float,default=0)
        self.parser.add_argument('--retina',type=int,default=1)
        self.isTrain = True
