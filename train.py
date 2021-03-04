import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from hparams import hparams
from models import Tacotron2
from dataset import TextMelLoader, TextMelCollate




device = 'cuda' 
torch.manual_seed(hparams.seed)
torch.cuda.manual_seed(hparams.seed)




class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, mel_out, mel_out_postnet, gate_out, mel_target, gate_target):
        # mel-rec
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_postnet, mel_target)

        # stop
        gate_target = gate_target.view(-1, 1)
        gate_out = gate_out.view(-1, 1)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        return mel_loss + gate_loss, mel_loss, gate_loss




def validate(model, loss_f, val_loader, epoch):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_mel_loss = 0.0
        val_gate_loss = 0.0
        num = len(val_loader)

        for _i, batch in enumerate(val_loader):
            # flow [1]:  data to 'cuda'
            text_padded, input_lengths, mel_padded, gate_padded, _output_lengths = batch
            text_padded = text_padded.to(device).long()
            input_lengths = input_lengths.to(device).long()
            mel_padded = mel_padded.to(device).float()
            gate_padded = gate_padded.to(device).float()


            # flow [2]:  predict
            mels_pre, mels_pre_postnet, gates_pre, _alignments_pre = model(text_padded, input_lengths, mel_padded)


            # flow [3]:  loss
            loss, mel_loss, gate_loss = loss_f(mels_pre, mels_pre_postnet, gates_pre, mel_padded, gate_padded)
            reduced_loss = loss.item() 
            reduced_mel_loss = mel_loss.item() 
            reduced_gate_loss = gate_loss.item() 


            # flow [4]:  add
            val_loss += reduced_loss
            val_mel_loss += reduced_mel_loss
            val_gate_loss += reduced_gate_loss



        val_loss = val_loss / num
        val_mel_loss = val_mel_loss / num
        val_gate_loss = val_gate_loss / num


    print("Validation loss {}: {:9f}  {:9f}  {:9f}  ".format(epoch, val_loss, val_mel_loss, val_gate_loss))

    model.train()
    return
        


def train_tacotron():
    # data
    trainset = TextMelLoader(hparams.mel_training_files, hparams)
    valset = TextMelLoader(hparams.mel_validation_files, hparams)

    train_loader = DataLoader(trainset, num_workers=0, shuffle=True, batch_size=hparams.batch_size, collate_fn=TextMelCollate, drop_last=True)
    val_loader = DataLoader(valset, num_workers=0, shuffle=False, batch_size=hparams.batch_size, collate_fn=TextMelCollate)


    # model
    model = Tacotron2(hparams).cuda()
    

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)


    # loss
    loss_f = Tacotron2Loss()
    

    # ckpt
    if hparams.checkpoint_path is not None:
        checkpoint_dict = torch.load(hparams.checkpoint_path)

        # dict
        model.load_state_dict(checkpoint_dict['state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        aready_epoch = checkpoint_dict['epoch']
    else:
        aready_epoch = 0
            


    # iteration
    model.train()
    iteration = 0
    for epoch in range(aready_epoch, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for _i, batch in enumerate(train_loader):
            # time
            start = time.perf_counter()


            # flow [1]:  data to 'cuda'
            text_padded, input_lengths, mel_padded, gate_padded, _output_lengths = batch
            # for train input
            text_padded = text_padded.to(device).long()
            input_lengths = input_lengths.to(device).long()
            mel_padded = mel_padded.to(device).float()
            # for loss
            gate_padded = gate_padded.to(device).float()


            # flow [2]:  predict
            mels_pre, mels_pre_postnet, gates_pre, _alignments_pre = model(text_padded, input_lengths, mel_padded)


            # flow [3]:  loss
            loss, mel_loss, gate_loss = loss_f(mels_pre, mels_pre_postnet, gates_pre, mel_padded, gate_padded)
            reduced_loss = loss.item() # loss
            _reduced_mel_loss = mel_loss.item() # only show
            _reduced_gate_loss = gate_loss.item() # only show


            # backward
            model.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)


            # apply
            optimizer.step()


            # logs
            print("Train loss {} {:.6f} {:.6f} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(iteration, reduced_loss, _reduced_mel_loss, _reduced_gate_loss, grad_norm, time.perf_counter() - start))
            iteration += 1
            
                
        # after 1 epoch
        validate(model, loss_f, val_loader, epoch) 
        torch.save({'epoch': iteration,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 
                os.path.join(hparams.output_directory, "checkpoint_{}".format(epoch)))




if __name__ == '__main__':
    train_tacotron()
