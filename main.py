import torch
from torch import nn
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import RepeatedStratifiedKFold

torch.manual_seed(111)
batch_size = 32

def loss_vs_epochs(epoch:list, loss_gen:list, loss_disc:list):
    plt.plot(epoch, loss_gen, lw=2, color='red', label='Generator')
    plt.plot(epoch, loss_disc, lw = 2, color='blue', label='Discriminator')
    plt.xlabel('EPOCH')
    plt.ylabel('Loss')
    plt.title('Loss vs EPOCHS')
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig('loss_epochs_curve.png', dpi=300)
    plt.show()

def gen_real_values(x):
    buffer = []
    for i in x:
        buffer.append([i, torch.sin(i)])
    t = pad_sequence([torch.tensor(x) for x in buffer], batch_first=True)
    return t

# metricas
def wmape(y_real, y_predicted):
    sum_y_real = torch.sum(y_real)
    for predicted_value, real_value in zip(y_predicted, y_real):
        nominator = (abs(real_value-predicted_value)/real_value) * 100 * real_value
    return float(nominator/sum_y_real)

def wape(y_real, y_predicted):
    sum_y_real = float(torch.sum(torch.abs(y_real)))
    nominator = 0
    for predicted_value, real_value in zip(y_predicted, y_real):
        nominator += abs(predicted_value - real_value)

    return float(nominator/sum_y_real)



def gen_train_data_set():
    train_data_length = 1024
    train_data = torch.zeros((train_data_length, 2))
    train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
    train_data[:, 1] = torch.sin(train_data[:, 0])
    train_labels = torch.zeros(train_data_length)
    train_set = [
        (train_data[i], train_labels[i]) for i in range(train_data_length)
    ]

    plot_args = [train_data[:, 0], train_data[:, 1], 'red', 'Real Data']
    plot(plot_args, None)
    return train_set


def get_train_loader(train_set):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    return train_loader



def plot(plot1, plot2):
    if plot2 != None:
        plt.plot(plot1[0], plot1[1], '.', c=plot1[2])
        plt.title(plot1[3])

        plt.plot(plot2[0], plot2[1], '.', c=plot2[2])
        plt.title(plot2[3])
        plt.show()
    else:
        plt.plot(plot1[0], plot1[1], '.', c=plot1[2])
        plt.title(plot1[3])
        plt.show()



# discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

# generator class
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

# training
def train_model(lr, num_epochs, train_loader, generator):



    loss_function = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    epoch_array = []
    gen_loss_array = []
    disc_loss_array = []
    for epoch in range(1, num_epochs + 1):
        for n, (real_samples, _) in enumerate(train_loader):
            # Data for training the discriminator
            real_samples_labels = torch.ones((batch_size, 1))
            latent_space_samples = torch.randn((batch_size, 2))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels)
            )

            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(
                output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, 2))

            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()

            # Test model every 50 epochs
            if epoch % 50 == 0 and n == batch_size - 1:
                calc_metric(generator, epoch)
                epoch_array.append(epoch)
                gen_loss_array.append(loss_generator.item())
                disc_loss_array.append(loss_discriminator.item())
                print(f"Loss D.: {loss_discriminator}")
                print(f"Loss G.: {loss_generator}")

    loss_vs_epochs(epoch_array, gen_loss_array, disc_loss_array)



def calc_metric(generator, e):
    latent_space_samples = torch.randn(100, 2)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.detach()
    plot_args_1 = [generated_samples[:, 0], generated_samples[:, 1], 'green', 'Gen. Data, Epochs: ' + str(e)]

    # calc metrics
    y_true = generated_samples.clone()
    #y_true[:, 0] = 2 * math.pi * torch.rand(100)
    y_true[:, 1] = torch.sin(y_true[:, 0])
    print('-----------------------------------------------------------')
    print('Model by Epoch', e)
    # mean squared error
    print('MSE:', mean_squared_error(y_true[:, 1], generated_samples[:, 1]))
    # mean absolute error
    print('MAE:', mean_absolute_error(y_true[:, 1], generated_samples[:, 1]))
    # R-squared
    print('R-squared:', r2_score(y_true[:, 1], generated_samples[:, 1]))
    # mean absolute percentage error
    print('MAPE:',mean_absolute_percentage_error(y_true[:, 1], generated_samples[:, 1]))
    # root mean squared error
    print('RMSE:', mean_squared_error(y_true[:, 1], generated_samples[:, 1], squared=False))
    # normalized root mean squared error
    print('NRMSE:', mean_squared_error(y_true[:, 1], generated_samples[:, 1], squared=False)/float(torch.mean(generated_samples[:, 1])))
    # wighted absoulte percentage error
    print('WAPE:', wape(y_true[:, 1], generated_samples[:, 1]))
    # wighted mean absoulte percentage error
    print('WMAPE:', wmape(y_true[:, 1], generated_samples[:, 1]))


    plot_args_2 = [y_true[:, 0], y_true[:, 1], 'red', 'Gen. Data, Epochs: ' + str(e)]
    plot(plot_args_1, plot_args_2)


if __name__ == "__main__":
    # data loading and generation
    train_set = gen_train_data_set()
    train_loader = get_train_loader(train_set)

    # discriminator and generator instances
    discriminator = Discriminator()
    generator = Generator()

    # training
    train_model(0.0001, 500, train_loader, generator)
















