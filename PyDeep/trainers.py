import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Trainer(object):
    """
    """


    def __init__(self, X, Y, model, criterion, optim, batch_size, epochs, train_size, pace_eval_test = 1, seed = 12) -> None:
        """
        """
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.pace_eval_test = pace_eval_test

        self.input_train, self.input_test, self.output_train, self.output_test = train_test_split(X, Y, random_state=seed, train_size=train_size)

        return None
    

    def batch(self, X, Y):
        """
        """

        np.random.seed(self.seed)

        N = X.shape[0]
        indices = np.arange(N)

        np.random.shuffle(indices)

        return [X[indices[start: start + self.batch_size]] for start in range(0, N, self.batch_size)], [Y[indices[start: start + self.batch_size]] for start in range(0, N, self.batch_size)]
        

    
    def training(self) -> tuple[list[float], list[float]]:
        """
        """
        Xb_train, Yb_train = self.batch(self.input_train, self.output_train)
        Xb_test, Yb_test = self.batch(self.input_test, self.output_test)

        loss_train, loss_test = [], []
        epoch_train, epoch_test = [], []

        for epoch in tqdm(range(self.epochs), desc="TRAINING"):
            
            epoch_loss_train, epoch_loss_test = 0, 0

            for xb_train, yb_train in zip(Xb_train, Yb_train):

                epoch_loss_train += self.criterion(self.model(xb_train), yb_train)
                self.model.backward(self.criterion.backward())
                self.optim.step()

            epoch_loss_train /= len(Xb_train)
            loss_train.append(epoch_loss_train)
            epoch_train.append(epoch+1)

            if not epoch % self.pace_eval_test:

                for xb_test, yb_test in zip(Xb_test, Yb_test):

                    epoch_loss_test += self.criterion(self.model(xb_test, track=False), yb_test, track=False)

                epoch_loss_test /= len(Xb_test)
                loss_test.append(epoch_loss_test)
                epoch_test.append(epoch+1)


        return epoch_train, loss_train, epoch_test, loss_test