import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import Tacotron2
from dataset import TextMelLoader

from hparams import hparams



torch.manual_seed(hparams.seed)
torch.cuda.manual_seed(hparams.seed)
device = 'cuda' # 先单核, 下一版再分布式



class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                   nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss, mel_loss, gate_loss




def validate(model, loss_f, val_loader, epoch):
    model.eval()
    with torch.no_grad():

        val_loss = 0.0
        val_mel_loss = 0.0
        val_gate_loss = 0.0
        num = len(val_loader)
        for i, batch in enumerate(val_loader):
            # flow [1]:  data to 'cuda'
            text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
            text_padded = text_padded.to(device).long()
            input_lengths = input_lengths.to(device).long()
            max_len = torch.max(input_lengths.data).item().to(device)
            mel_padded = mel_padded.to(device).float()
            gate_padded = gate_padded.to(device).float()
            output_lengths = output_lengths.to(device).long()

            x, y = (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded),
            )


            # flow [2]:  predict
            y_pred = model(x)


            # flow [3]:  loss
            loss, mel_loss, gate_loss = loss_f(y_pred, y)
            reduced_val_loss = loss.item()
            reduced_val_mel_loss = mel_loss.item()
            reduced_val_gate_loss = gate_loss.item()


            # add
            val_loss += reduced_val_loss
            val_mel_loss += reduced_val_mel_loss
            val_gate_loss += reduced_val_gate_loss



        val_loss = val_loss / num
        val_mel_loss = val_mel_loss / num
        val_gate_loss = val_gate_loss / num


    print("Validation loss {}: {:9f}  ".format(epoch, val_loss))
        # logger.log_validation(val_loss, val_mel_loss, val_gate_loss, model, y, y_pred, iteration)

    model.train()
    return
        


def train_tacotron():
    # data
    trainset = TextMelLoader(hparams.mel_training_files, hparams)
    valset = TextMelLoader(hparams.mel_validation_files, hparams)

    train_loader = DataLoader(trainset, num_workers=0, shuffle=True,
                              batch_size=hparams.batch_size,
                              drop_last=True)
    val_loader = DataLoader(valset, num_workers=0,
                                shuffle=False, batch_size=hparams.batch_size)


    # model, 分布式的下一版再写
    model = Tacotron2(hparams).cuda()
    

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)


    # loss
    loss_f = Tacotron2Loss()
    

    # ckpt
    if hparams.checkpoint_path is not None:
        checkpoint_path = hparams.checkpoint_path
        checkpoint_dict = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_dict['state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        aready_epoch = checkpoint_dict['epoch']
        # learning_rate = checkpoint_dict['learning_rate']
        

        # lr的衰减先放在这里
        # if hparams.use_saved_learning_rate:
        #     learning_rate = _learning_rate
    else:
        aready_epoch = 0
            

    # iteration
    model.train()
    iteration = 0
    for epoch in range(aready_epoch, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            # time
            start = time.perf_counter()
            # lr的衰减先放在这里
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = learning_rate


            # flow [1]:  data to 'cuda'
            text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
            text_padded = text_padded.to(device).long()
            input_lengths = input_lengths.to(device).long()
            max_len = torch.max(input_lengths.data).item().to(device)
            mel_padded = mel_padded.to(device).float()
            gate_padded = gate_padded.to(device).float()
            output_lengths = output_lengths.to(device).long()

            x, y = (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded),
            )


            # flow [2]:  predict
            y_pred = model(x)


            # flow [3]:  loss
            loss, mel_loss, gate_loss = loss_f(y_pred, y)
            reduced_loss = loss.item() # loss
            reduced_mel_loss = mel_loss.item() # only show
            reduced_gate_loss = gate_loss.item() # only show


            # backward
            model.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh)


            # apply
            optimizer.step()


            # logs
            print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(iteration, reduced_loss, grad_norm, time.perf_counter() - start))
            iteration += 1
            
                
        # after 1 epoch
        validate(model, loss_f, val_loader, epoch) 
        torch.save({'epoch': iteration,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 
                os.path.join(hparams.output_directory, "checkpoint_{}".format(epoch)))



if __name__ == '__main__':
    # 不懂, 先放在这里
    # torch.backends.cudnn.enabled = hparams.cudnn_enabled
    # torch.backends.cudnn.benchmark = hparams.cudnn_benchmark


    train_tacotron()
