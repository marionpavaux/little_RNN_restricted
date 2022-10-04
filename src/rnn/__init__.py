from os.path import dirname, abspath, join
import pandas as pd 

ROOT_DIR = dirname(abspath(__file__))
PATH = abspath(join(ROOT_DIR, '../../tests_results/'))

# Training
LR =  "RESTRICTED"
BATCH_SIZE = ["RESTRICTED","RESTRICTED"] 
GPU = "cuda:2"
ALPHA = pd.DataFrame({20: ["RESTRICTED"], 40: ["RESTRICTED"], 60: ["RESTRICTED"], 80: ["RESTRICTED"], 100: ["RESTRICTED"] ,120: ["RESTRICTED"]})
WORTH_MP = 200
TEST_SIZE = "RESTRICTED"


from .save import load_data, load_checkpoint, dump_checkpoint, dump_loss
from .predict import predict
from .train import train, train_per_configs
from .interpret import plot, plot_pred, plot_alpha_activity, plot_EI_activity, plot_alpha_distribution, plot_eigenvalues, plot_heatmap, plot_PCA, plot_loss_trajectory
from .models import RNN, LSTM, LSTM2, GRU, ARNN, AEIRNN, ASRNN, ASEIRNN, AFRNN, AFEIRNN, ASFRNN, ASFEIRNN


