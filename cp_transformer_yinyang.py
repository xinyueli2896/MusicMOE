import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.roformer_inject import RoFormerEncoder as RoFormerEncoderInject
from cp_transformer import RoFormerSymbolicTransformer, FramedDataset, fill_with_neg_inf, N_NORMAL_TOKENS, N_TOKENS, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from peft import LoraConfig, get_peft_model
from modules.yinyang_cross_attn import LowRankMultiheadAttention
import sys

TRAIN_LENGTH = 384
MAX_STEPS = 40000

def end_generator(generator):
    try:
        next(generator)
        raise ValueError('Generator did not end')
    except StopIteration as e:
        return e.value


class RoFormerSymbolicTransformerInjected(RoFormerSymbolicTransformer):

    def __init__(self, large=False):
        super().__init__(large=large)

    def get_base_model(self, config):
        return RoFormerEncoderInject(config)

    def forward(self, x):
        # x: [batch, seq, subseq]
        # Use local encoder to encode subsequences
        batch_size, seq_len, subseq_len = x.shape
        h, emb = self.local_encode(x)
        h = h.view(batch_size, seq_len, -1)
        # Prepend SOS token and remove the last token
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h[:, :-1]], dim=1)
        # Use global transformer to decode
        h = (yield from self.model(h, attention_mask=self.buffered_future_mask(h)))[0]
        return self.local_decode(h, emb)


    def global_sampling(self, x, max_seq_len=384, temperature=1.0):
        batch_size, seq_len, subseq_len = x.shape
        h, _ = self.local_encode(x)
        h = h.view(batch_size, seq_len, -1)
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h], dim=1)
        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list
        for i in range(seq_len + 1, max_seq_len):
            yield ['generation_step', i]
            if i % 10 == 0:
                print('Sampling', i, '/', max_seq_len)
            h_out = (yield from self.model(h, attention_mask=self.buffered_future_mask(h)))[0]
            y_next = self.local_sampling(h_out[:, -1], temperature=temperature)
            y.append(y_next)
            h = torch.cat([h, self.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)], dim=1)
        return y

    def loss(self, x, pitch_shift):
        x = self.preprocess(x, pitch_shift)
        y = yield from self(x)
        return F.cross_entropy(y.view(-1, N_TOKENS), x.view(-1), ignore_index=PAD_TOKEN)

class RoformerWithRoLA(L.LightningModule):

    def __init__(self, model_fp, large=False):
        super(RoformerWithRoLA, self).__init__()
        base_model = RoFormerSymbolicTransformerInjected.load_from_checkpoint(model_fp, large=large)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
        )
        self.wrapped_model = get_peft_model(base_model, lora_config)

    def loss(self, x, pitch_shift):
        return (yield from self.wrapped_model.loss(x, pitch_shift))

class RoformerYinyang(L.LightningModule):

    def __init__(self, model_fp, large=False):
        super().__init__()
        self.save_hyperparameters()
        if not os.path.exists(model_fp):
            # Use relative path
            model_fp = os.path.join('ckpt', os.path.basename(model_fp))
        base_model1 = RoFormerSymbolicTransformerInjected.load_from_checkpoint(model_fp, large=large)
        base_model2 = RoFormerSymbolicTransformerInjected.load_from_checkpoint(model_fp, large=large)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
        )
        self.wrapped_model1 = get_peft_model(base_model1, lora_config)
        self.wrapped_model2 = get_peft_model(base_model2, lora_config)
        self.n_layers = base_model1.num_layers
        self.yinyang_attn = nn.ModuleList([
            LowRankMultiheadAttention(
                in_dim=base_model1.hidden_size,
                embed_dim=256,
                num_heads=4,
                dropout=0.5,
            )
            for _ in range(self.n_layers // 2)
        ])

    def forward(self, x1, x2):
        gen1 = self.wrapped_model1(x1)
        gen2 = self.wrapped_model2(x2)
        for layer in range(self.n_layers):
            data1 = next(gen1); assert data1[0] == 'hidden_states'
            data2 = next(gen2); assert data2[0] == 'hidden_states'
            if layer % 2 == 0:
                yinyang_weights = self.yinyang_attn[layer // 2](data2[1], data1[1], data1[1], None)
            data1 = next(gen1); assert data1[0] == 'prenorm_output'
            data2 = next(gen2); assert data2[0] == 'prenorm_output'
            if layer % 2 == 0:
                data2[1] = data2[1] + yinyang_weights
        return end_generator(gen2)

    def global_sampling(self, x1, x2, temperature=1.0):
        print('Yinyang Sampling')
        batch_size, max_seq_len, subseq_len = x1.shape
        seq_len = x2.shape[1]
        gen2 = self.wrapped_model2.global_sampling(x2, max_seq_len=max_seq_len, temperature=temperature)
        for i in range(seq_len + 1, max_seq_len):
            data2 = next(gen2); assert data2[0] == 'generation_step'
            gen1 = self.wrapped_model1(x1)
            for layer in range(self.n_layers):
                data1 = next(gen1); assert data1[0] == 'hidden_states'
                data2 = next(gen2); assert data2[0] == 'hidden_states'
                if layer % 2 == 0:
                    yinyang_weights = self.yinyang_attn[layer // 2](data2[1], data1[1], data1[1], None)
                data1 = next(gen1); assert data1[0] == 'prenorm_output'
                data2 = next(gen2); assert data2[0] == 'prenorm_output'
                if layer % 2 == 0:
                    data2[1] = data2[1] + yinyang_weights
        return end_generator(gen2)

    def preprocess(self, x, pitch_shift):
        batch_size, seq_len, subseq_len = x.shape
        x = x.view(batch_size, seq_len * 2, subseq_len // 2)
        x_processed = self.wrapped_model1.preprocess(x, pitch_shift)
        x_processed = x_processed.view(batch_size, seq_len, 2, -1)
        return x_processed[:, :, 0, :], x_processed[:, :, 1, :]

    def loss(self, x, pitch_shift):
        x1, x2 = self.preprocess(x, pitch_shift)
        y = self(x1, x2)
        return F.cross_entropy(y.view(-1, N_TOKENS), x2.reshape(-1), ignore_index=PAD_TOKEN)


    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        max_lr = 1e-4
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr)
        return optimizer

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        # Get the full state dict from the parent class
        full_state_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Filter to keep only trainable parameters
        trainable_state_dict = {
            k: v for k, v in full_state_dict.items()
            if self._is_trainable_param(k)
        }
        return trainable_state_dict

    def _is_trainable_param(self, param_name):
        # Check if the parameter is trainable
        for name, param in self.named_parameters():
            if name == param_name and param.requires_grad:
                return True
        return False

def sanity_check():
    # Do some sanity check
    ckpt_path = 'ckpt/cp_transformer_v0.3_small_batch_18.epoch=00.val_loss=0.48485.ckpt'
    model = RoformerWithRoLA(ckpt_path)
    model.cpu()
    x = [
        [
            [52, 34, 11, 254, 255, 255],
            [12, 34, 11, 34, 9, 5],
            [12, 34, 11, 34, 12, 7]
        ],
        [
            [52, 34, 11, 254, 255, 255],
            [12, 34, 7, 34, 12, 5],
            [254, 255, 255, 255, 255, 255]
        ]
    ]
    x = torch.tensor(x, dtype=torch.long)
    pitch_shift = torch.tensor([0, 0], dtype=torch.long)
    loss_generator = model.loss(x, pitch_shift)
    try:
        while True:
            internal_msg = next(loss_generator)
            print('Internal', internal_msg)
            if internal_msg[0] == 'hidden_states':
                internal_msg[1][0, :] = np.nan
    except StopIteration as e:
        loss_value = e.value
        print('Loss', loss_value)


if __name__ == '__main__':
    batch_size = int(sys.argv[1])
    fp_path = str(sys.argv[2])
    dataset_name = str(sys.argv[3])
    model_size = 'large' if 'large' in fp_path else 'small'
    n_gpus = max(torch.cuda.device_count(), 1)
    model_name = f'cp_transformer_yinyang_v1.23_{model_size}_batch_{batch_size * n_gpus}_{dataset_name}'
    net = RoformerYinyang(fp_path, model_size == 'large')
    train_set_loader = DataLoader(FramedDataset(f'data/{dataset_name}.pt', TRAIN_LENGTH, batch_size, split='train'), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset(f'data/{dataset_name}.pt', TRAIN_LENGTH, batch_size, split='val'), batch_size=None, num_workers=1, persistent_workers=True)
    checkpoint_callback = L.callbacks.ModelCheckpoint(monitor='val_loss',
                                                      save_top_k=10,
                                                      save_last=True,
                                                      dirpath=f'ckpt/{model_name}',
                                                      filename=model_name + '.{epoch:02d}.{val_loss:.5f}')

    # load from checkpoint
    checkpoint_path = None
    if len(sys.argv) > 4:
        checkpoint_path = sys.argv[4]
    trainer = L.Trainer(devices=n_gpus,
                        precision="bf16-mixed" if torch.cuda.is_available() else 32,
                        max_steps=MAX_STEPS,
                        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        callbacks=[checkpoint_callback],
                        val_check_interval=500,
                        limit_val_batches=25,
                        check_val_every_n_epoch=None,
                        logger=TensorBoardLogger("tb_logs", name=model_name),
                        strategy='auto' if n_gpus == 1 else 'ddp')
    net.strict_loading = False
    trainer.fit(net, train_set_loader, val_set_loader, ckpt_path=checkpoint_path)
    # save the model (parameters only)
    torch.save(net.state_dict(), f'ckpt/{model_name}.pt')