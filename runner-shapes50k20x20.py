from utils import AttributeDict
from tagger_exp import TaggerExperiment

p = AttributeDict()

p.encoder_proj = (2000, 1000, 500)
p.input_noise = 0.2
p.class_cost_x = 0
p.zhat_init_value = 0.26  # mean of the input data.

p.n_iterations = 3
p.n_groups = 4
p.lr = 0.0004
p.seed = 10
p.num_epochs = 100
p.batch_size = 100
p.valid_batch_size = 100

p.dataset = 'shapes50k20x20'
p.input_type = 'binary'

p.save_to = 'shapes50k20x20'

if __name__ == '__main__':
    experiment = TaggerExperiment(p)
    experiment.train()
