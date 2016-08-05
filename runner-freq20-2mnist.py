import sys
from utils import AttributeDict
from tagger_exp import TaggerExperiment

p = AttributeDict()

p.encoder_proj = (3000, 2000, 1000)
p.input_noise = 0.2
p.class_cost_x = 0.
p.zhat_init_value = 0.5

p.n_iterations = 3
p.n_groups = 4
p.lr = 0.001
p.labeled_samples = 1000
p.save_freq = 50
p.seed = 1
p.num_epochs = 150
p.batch_size = 100
p.valid_batch_size = 100
p.objects_per_sample = 2

p.dataset = 'freq20-2mnist'
p.input_type = 'continuous'

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--pretrain':
        p.save_to = 'freq20-2mnist-pretraining'
        experiment = TaggerExperiment(p)
        experiment.train()
    elif len(sys.argv) == 3 and sys.argv[1] == '--continue':
        p.load_from = sys.argv[2]
        p.save_to = 'freq20-2mnist-supervision'
        p.num_epochs = 50
        p.n_iterations = 4
        p.encoder_proj = (3000, 2000, 1000, 500, 250, 11)
        p.lr = 0.0002
        p.input_noise = 0.18
        p.class_cost_x = 0.1
        experiment = TaggerExperiment(p)
        experiment.train()
