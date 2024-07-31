import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_files.config import load_args
from dataloaders import get_dataloaders, loadModel
from trainer import Trainer
from tester import Tester

if __name__ == "__main__":
    args = load_args("program execution")

    model = loadModel(args)

    train_loader, val_loader, test_loader = get_dataloaders(args)

    trainer = Trainer(model, train_loader, val_loader, args)
    tester = Tester(model, test_loader, args)

    print('TRAINING START')
    trainer.run()
    print('TESTING START')
    tester.run()
