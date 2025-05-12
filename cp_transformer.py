import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roformer.modeling_roformer import RoFormerModel, RoFormerConfig, RoFormerEncoder
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import sys

TRAIN_LENGTH = 384
MAX_STEPS = 1000000

N_NORMAL_TOKENS = 3328
N_TOKENS = N_NORMAL_TOKENS + 3
SOS_TOKEN = N_NORMAL_TOKENS
EOS_TOKEN = N_NORMAL_TOKENS + 1
PAD_TOKEN = N_NORMAL_TOKENS + 2

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

class RoFormerSymbolicTransformer(L.LightningModule):

    def __init__(self, large=False):
        super().__init__()
        self.hidden_size = 768 if large else 512
        self.num_layers = 12 if large else 6
        self.num_attention_heads = 12 if large else 8
        self.intermediate_size = 3072 if large else 1024
        self.local_model_num_layers = 3
        self.local_model_num_attention_heads = 8
        self.local_model_intermediate_size = 768
        main_roformer_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.model = self.get_base_model(main_roformer_config)
        local_encoder_config = local_decoder_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.local_embedding = nn.Embedding(N_TOKENS, self.hidden_size)
        self.local_encoder = RoFormerEncoder(local_encoder_config)
        self.local_decoder = RoFormerEncoder(local_decoder_config)
        self.final_decoder = nn.Linear(self.hidden_size, N_TOKENS)
        self.global_sos = nn.Parameter(torch.randn(self.hidden_size))
        self._future_mask = torch.empty(0)

    def get_base_model(self, config):
        return RoFormerEncoder(config)

    def local_encode(self, x):
        batch_size, seq_len, subseq_len = x.shape
        x = x.view(-1, subseq_len)
        # prepend a <sos> token
        x = torch.cat([torch.full((x.shape[0], 1), SOS_TOKEN, dtype=torch.long, device=x.device), x], dim=-1)
        mask = x != PAD_TOKEN
        emb = self.local_embedding(x)
        h = self.local_encoder(emb, encoder_attention_mask=mask)[0]
        # get representation of the first token
        return h[:, 0], emb[:, :-1]

    def local_decode(self, h, emb):
        batch_size, subseq_len, _ = emb.shape
        # Add h as the first token of emb
        h = h.view(batch_size, 1, -1)
        emb = torch.cat([h, emb[:, 1:]], dim=1)
        # Create an autoregressive mask

        h = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
        return self.final_decoder(h)

    def local_sampling(self, h, max_subseq_len=32, temperature=1.0):
        batch_size, _ = h.shape
        y = torch.zeros((batch_size, 0), dtype=torch.long, device=h.device)
        emb = h[:, None, :]
        eos_triggered = torch.zeros(batch_size, dtype=torch.bool, device=h.device)
        for i in range(max_subseq_len):
            h = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
            if temperature == 0:
                p = F.one_hot(self.final_decoder(h).argmax(dim=-1), N_TOKENS).float()
            else:
                p = F.softmax(self.final_decoder(h[:, -1]) / temperature, dim=-1)
            y_next = torch.multinomial(p, 1)
            y_next[eos_triggered, :] = PAD_TOKEN  # If EOS has been triggered, pad the rest
            eos_triggered = eos_triggered | (y_next.squeeze(1) == EOS_TOKEN)
            y = torch.cat([y, y_next], dim=1)
            if torch.all(eos_triggered):
                break
            emb = torch.cat([emb, self.local_embedding(y_next)], dim=1)
        return y

    def global_sampling(self, x, max_seq_len=384, temperature=1.0):
        batch_size, seq_len, subseq_len = x.shape
        h, _ = self.local_encode(x)
        h = h.view(batch_size, seq_len, -1)
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h], dim=1)
        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list
        for i in range(seq_len + 1, max_seq_len):
            if i % 10 == 0:
                print('Sampling', i, '/', max_seq_len)
            h_out = self.model(h, attention_mask=self.buffered_future_mask(h))[0]
            y_next = self.local_sampling(h_out[:, -1], temperature=temperature)
            y.append(y_next)
            h = torch.cat([h, self.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)], dim=1)
        return y

    def buffered_future_mask(self, tensor):
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

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
        h = self.model(h, attention_mask=self.buffered_future_mask(h))[0]
        return self.local_decode(h, emb)


    def preprocess(self, x, pitch_shift):
        batch_size, seq_length, subseq_length = x.shape
        x = x.long().view(batch_size, seq_length, subseq_length // 3, 3)
        x_processed = torch.zeros(batch_size, seq_length, subseq_length // 3, 2, dtype=torch.long, device=x.device)
        pad_indices = x[:, :, :, 1] == 255
        eos_indices = x[:, :, :, 0] == 254
        is_not_drum = x[:, :, :, 0] != 127
        x_processed[:, :, :, 0] = x[:, :, :, 0]
        x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2] + 1) * 128 + pitch_shift[:, None, None] * is_not_drum
        x_processed[pad_indices] = PAD_TOKEN
        x_processed[:, :, :, 0][eos_indices] = EOS_TOKEN
        return x_processed.view(batch_size, seq_length, subseq_length // 3 * 2)
    def loss(self, x, pitch_shift):
        x = self.preprocess(x, pitch_shift)
        y = self(x)
        return F.cross_entropy(y.view(-1, N_TOKENS), x.view(-1), ignore_index=PAD_TOKEN)

    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('train_loss', loss)
        # scheduler step
        scheduler = self.lr_schedulers()
        scheduler.step()
        self.log('training/lr', scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        max_lr = 1e-4
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=MAX_STEPS, pct_start=0.005)
        return [optimizer], [scheduler]

class FramedDataset(IterableDataset):

    def __init__(self, file_path, target_length, batch_size, split='all', split_ratio=10):
        self.file_path = file_path
        self.length = torch.load(file_path[:-3] + '.length.pt', weights_only=True)
        self.start = torch.cumsum(self.length, dim=0) - self.length
        # Invalid samples are those whose length is less than min_length
        is_valid = self.length >= target_length
        self.valid_indices = torch.arange(len(self.start))[is_valid]
        # Get training or validation split
        if split == 'all':
            pass
        elif split == 'train':
            self.valid_indices = self.valid_indices[self.valid_indices % split_ratio != 0]
        elif split == 'val':
            self.valid_indices = self.valid_indices[self.valid_indices % split_ratio == 0]
        self.split = split
        self.valid_song_count = len(self.valid_indices)
        self.target_length = target_length
        self.batch_size = batch_size
        print('Metadata for dataset', file_path, 'loaded. Number of valid songs:', self.valid_song_count)

    def __iter__(self):
        data = torch.load(self.file_path, weights_only=True)
        pitch_shift_range = torch.load(self.file_path[:-3] + '.pitch_shift_range.pt', weights_only=True).reshape(-1, 2)
        pitch_shift_range[pitch_shift_range[:, 0] < -5, 0] = -5
        pitch_shift_range[pitch_shift_range[:, 1] > 6, 1] = 6
        if self.split == 'val':
            pitch_shift_range = torch.zeros_like(pitch_shift_range)  # No pitch shift for validation
        print('Data for dataset', self.file_path, 'loaded.')
        while True:
            indices = torch.randperm(len(self.valid_indices))
            for i in range(0, len(self.valid_indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_pitch_shift_range = pitch_shift_range[self.valid_indices[batch_indices]]
                raw_ids = self.valid_indices[batch_indices]
                # starts = torch.randint(self.start[raw_ids], self.start[raw_ids] + self.length[raw_ids] - self.target_length)
                starts = torch.floor(torch.rand(len(raw_ids)) * (self.length[raw_ids] - self.target_length)).long() + self.start[raw_ids]
                index_matrix = torch.arange(self.target_length).view(1, -1) + starts.view(-1, 1)
                # Shift the pitch in range [min, max], inclusive
                batch_pitch_shift = torch.floor(torch.rand(len(raw_ids)) * (batch_pitch_shift_range[:, 1] - batch_pitch_shift_range[:, 0] + 1)).long() + batch_pitch_shift_range[:, 0]
                yield data[index_matrix], batch_pitch_shift

def sanity_check():
    # Do some sanity check
    model = RoFormerSymbolicTransformer()
    x = [
        [
            [52, 34, 12, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN],
            [12, 34, 52, 34, 12, EOS_TOKEN],
            [12, 34, 52, 34, 12, 52]
        ],
        [
            [52, 34, 12, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN],
            [12, 34, 52, 34, 12, EOS_TOKEN],
            [EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]
        ]
    ]
    x = torch.tensor(x, dtype=torch.long)
    loss = model.loss(x)
    print(loss)

if __name__ == '__main__':
    batch_size = int(sys.argv[1])
    model_size = str(sys.argv[2])
    assert model_size in ['small', 'large']
    n_gpus = max(torch.cuda.device_count(), 1)
    model_name = f'cp_transformer_v0.3_{model_size}_batch_{batch_size * n_gpus}_schedule'
    net = RoFormerSymbolicTransformer(model_size == 'large')
    train_set_loader = DataLoader(FramedDataset('data/rwc_cp16_v1.pt', TRAIN_LENGTH, batch_size), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset('data/rwc_cp16_v1.pt', TRAIN_LENGTH, batch_size), batch_size=None, num_workers=1, persistent_workers=True)
    checkpoint_callback = L.callbacks.ModelCheckpoint(monitor='val_loss',
                                                      save_top_k=10,
                                                      save_last=True,
                                                      dirpath=f'ckpt/{model_name}',
                                                      filename=model_name + '.{epoch:02d}.{val_loss:.5f}')

    # load from checkpoint
    checkpoint_path = None
    if len(sys.argv) > 3:
        checkpoint_path = sys.argv[3]
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
    trainer.fit(net, train_set_loader, val_set_loader, ckpt_path=checkpoint_path)
    # save the model (parameters only)
    torch.save(net.state_dict(), f'ckpt/{model_name}.pt')