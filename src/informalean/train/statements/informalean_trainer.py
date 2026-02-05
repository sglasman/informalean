from transformers import AutoModelForCausalLM
from trl import SFTTrainer
from torch.utils.data import WeightedRandomSampler


class InformaleanStatementTrainer(SFTTrainer):

    def __init__(self, *args, weights, **kwargs):
        super(InformaleanStatementTrainer, self).__init__(*args, **kwargs)
        self.weights = weights

    def _get_train_sampler(self, train_dataset):
        return WeightedRandomSampler(
            weights=self.weights,
            num_samples=len(train_dataset),
            replacement=True,
        )