import pandas as pd
import torch.nn.functional as F
import os
from tabulate import tabulate
from evaluation import evaluation


class Tester:
    def __init__(self, model, test_loader, args):
        self.model = model
        self.test_loader = test_loader
        self.args = args
        self.loss_fn = F.cross_entropy
        self.results_dir = f'results/{args.model}'
        self.results_file = f"{args.model}_test_results.csv"

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def run(self):
        results = evaluation(self.model, self.test_loader, self.loss_fn)
        test_df = pd.DataFrame({'test_loss': results["average_loss"],
                                'accuracy': results["accuracy"]}, index=[0])
        print(tabulate(test_df, headers='keys', tablefmt='psql', showindex=False))
        test_df.to_csv(os.path.join(self.results_dir, self.results_file), index=False)
        print(f'Test results file: {self.results_file}')
