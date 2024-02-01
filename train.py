from preprocess import load_data
from dataset import CustomDataset
from model import DumbNetwork
from draw_topo import draw_topo
import copy
import yaml
import torch
import torch.nn as nn
import math
from tensorboardX import SummaryWriter
class Train():
    def __init__(self):
        with open('hyper_params.yml', "r") as file:
            args = yaml.load(file, Loader=yaml.FullLoader)
        self.args = args
        self.data,self.label = load_data(self.args['dataset_name'])
        self.writer = SummaryWriter()
    def loso(self):
        csub = []
        # cross-validation, LOSO
        for session_id_main in range(3):
            for subject_id_main in range(15):
                csub.append(self.cross_subject(session_id_main, subject_id_main))

    def cross_subject(self,session_id, subject_id):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        one_session_data, one_session_label = copy.deepcopy(self.data[session_id]), copy.deepcopy(self.label[session_id])
        train_idxs = list(range(15))
        del train_idxs[subject_id]
        test_idx = subject_id
        target_data, target_label = copy.deepcopy(one_session_data[test_idx]), copy.deepcopy(
            one_session_label[test_idx])
        source_data, source_label = copy.deepcopy(one_session_data[train_idxs]), copy.deepcopy(
            one_session_label[train_idxs])
        del one_session_label
        del one_session_data

        source_loaders = []
        for j in range(len(source_data)):
            source_loaders.append(
                torch.utils.data.DataLoader(dataset=CustomDataset(source_data[j], source_label[j]),
                                            batch_size=self.args['batch_size'],
                                            shuffle=True,
                                            drop_last=True))
        target_loader = torch.utils.data.DataLoader(dataset=CustomDataset(target_data, target_label),
                                                    batch_size=self.args['batch_size'],
                                                    shuffle=True,
                                                    drop_last=True)
        model = DumbNetwork()
        model.to(device)
        source_iters = []
        for i in range(len(source_loaders)):
            source_iters.append(iter(source_loaders[i]))
        target_iter = iter(target_loader)
        epoch = self.args['epoch']
        batch_size = self.args['batch_size']
        dataset_name = self.args['dataset_name']
        if dataset_name == 'seed3':
            iteration = math.ceil(epoch * 3394 / batch_size)
        elif dataset_name == 'seed4':
            iteration = math.ceil(epoch * 820 / batch_size)
        else:
            iteration = 5000
        log_interval = self.args['log_interval']
        criterion = nn.CrossEntropyLoss().to(device)

        for i in range(1, iteration+1):
            model.train()
            # LEARNING_RATE = self.lr / math.pow((1 + 10 * (i - 1) / (self.iteration)), 0.75)
            LEARNING_RATE = self.args['learning_rate']
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            for j in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[j])
                except Exception as err:
                    source_iters[j] = iter(source_loaders[j])
                    source_data, source_label = next(source_iters[j])
                try:
                    target_data, _ = next(target_iter)
                except Exception as err:
                    target_iter = iter(target_loader)
                    target_data, _ = next(target_iter)
                source_data, source_label = source_data.to(device), source_label.squeeze().to(device)
                target_data = target_data.to(device)

                optimizer.zero_grad()
                output = model(source_data)

                # calculate CrossEntropyLoss
                loss = criterion(output, source_label.long())
                self.writer.add_scalar('loss', loss.item(), i)
                loss.backward()
                optimizer.step()
                print('Train source' + str(j) + ', iter: {} [({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item()))
                if i % log_interval == 0:
                    fig = self.get_topo(model)
                    self.writer.add_figure('topo', fig, i)
        return

    def hook_forward(self,module, input, output):
        # print('shape of input:', input[0].shape)
        # print('shape of output:', output.data.shape)
        print('forward')

    def hook_backward_grad(self,module, grad_in, grad_out):
        # print('shape of grad_in:', grad_in[0].shape)
        # print('shape of grad_out:', grad_out[0].shape)
        print('backward')
    def get_topo(self,model):
        weight = model.fc1.weight.data.to('cpu').numpy()
        fig = draw_topo(weight)
        return fig
