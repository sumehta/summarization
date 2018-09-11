import os
import pandas as pd
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class TextAbstraction(text_problems.Text2TextProblem):
  """Predict next line of poetry from the last line. From Gutenberg texts."""

  @property
  def approx_vocab_size(self):
    return 50000  # ~8k

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return True

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 30,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 10,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
      if dataset_split == problem.DatasetSplit.TRAIN:
          import pdb; pdb.set_trace()
          train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
          for idx, row in train_data.iterrows():
              yield {
              "inputs": row['extractor_op'],
              "targets": row['ground_truth']
              }
      elif dataset_split == problem.DatasetSplit.EVAL:
          import pdb; pdb.set_trace();
          val_data = pd.read_csv(os.path.join(data_dir, 'val.csv'))
          for idx, row in val_data.iterrows():
              yield {
              "inputs": row['extractor_op'],
              "targets": row['ground_truth']
              }
