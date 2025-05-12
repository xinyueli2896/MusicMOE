import numpy as np

from cp_transformer_yinyang import RoformerYinyang, SOS_TOKEN, EOS_TOKEN
from preprocess_large_midi_dataset import preprocess_midi, DURATION_TEMPLATES
from cp_transformer_inference import decode_output, decompress
from settings import RWC_DATASET_PATH
import torch
import pretty_midi
import os

def cond_continuation(model, midi_path, prompt_length=100, generation_length=384, temperature=1.0, n_samples=1):
    if 'drums_nondrum' in model.save_name:
        ins_ids = ['drum', 'nondrum']
    elif 'nondrum_drums' in model.save_name:
        ins_ids = ['nondrum', 'drum']
    elif '_chords' in model.save_name:
        ins_ids = ['from-1', 'upto-0']
    else:
        ins_ids = ['track-1', 'track-0'] if '_chord_mel' in model.save_name else ['track-0', 'track-1']
    x1, x2 = decompress(model, preprocess_midi(midi_path, 16, ins_ids=ins_ids)[0])
    print(x1.shape, x2.shape)
    x1 = x1[:, :generation_length]
    x2 = x2[:, :prompt_length]
    cond_notes = [x1[:, i, :] for i in range(x1.shape[1])]
    decode_output((cond_notes, [x2[:, i, :] for i in range(x2.shape[1])]),
                  f'temp/{model.save_name}/{os.path.basename(midi_path)}_prompt.mid')
    with torch.no_grad():
        x1 = x1.repeat(n_samples, 1, 1)
        x2 = x2.repeat(n_samples, 1, 1)
        output = model.global_sampling(x1, x2, temperature=temperature)
    for i in range(n_samples):
        output_i = [output[j][i:i + 1, :] for j in range(len(output))]
        decode_output((cond_notes, output_i), f'temp/{model.save_name}/{os.path.basename(midi_path)}_temp{temperature}_continuation_{i}.mid')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python cp_transformer_yinyang_inference.py <model_path>')
        exit(1)
    model_path = sys.argv[1]
    model = RoformerYinyang.load_from_checkpoint(model_path, strict=False)
    model.save_name = os.path.basename(model_path)
    model.cuda()
    model.eval()
    if 'nondrum_drums' in model.save_name:
        cond_continuation(model, 'input/RM-P003.SMF_SYNC.MID', temperature=1.0, generation_length=384, n_samples=8, prompt_length=48)
    elif '_chord_mel' in model.save_name:
        cond_continuation(model, 'input/RM-P003.simple.MID', temperature=1.0, generation_length=384, n_samples=8, prompt_length=48)
        cond_continuation(model, 'input/ashover5.mid', temperature=1.0, generation_length=384, n_samples=8, prompt_length=48)
        cond_continuation(model, 'input/ashover2.mid', temperature=1.0, generation_length=384, n_samples=8, prompt_length=48)
    elif '_chords' in model.save_name:
        # cond_continuation(model, 'temp/la_beat/f947e58c78aa7c8055ef8dfc424ca22e_beat.mid', temperature=1.0, generation_length=384, n_samples=8, prompt_length=16)
        cond_continuation(model, 'temp/la_beat/d9520bbf2bccd6424aa09f5694aa68f7_beat.mid', temperature=1.0, generation_length=384, n_samples=8, prompt_length=16)
        cond_continuation(model, 'temp/la_beat/cf5f3bc804e474f4d0baf0c74656b042_beat.mid', temperature=1.0, generation_length=384, n_samples=8, prompt_length=16)
        # cond_continuation(model, 'temp/la_beat/394045b83f247bb862d7b09b1aacd78f_beat.mid', temperature=1.0, generation_length=384, n_samples=8, prompt_length=16)
        # cond_continuation(model, 'temp/la_beat/4261342f0970488e1381cb39867c48e1_beat.mid', temperature=1.0, generation_length=384, n_samples=8, prompt_length=16)
    else:
        cond_continuation(model, 'input/RM-P003.simple.MID', temperature=1.0, generation_length=384, n_samples=8, prompt_length=48)
        cond_continuation(model, 'input/Ode-To-Joy.mid', temperature=1.0, generation_length=384, n_samples=8, prompt_length=48)
        cond_continuation(model, 'input/ashover5.mid', temperature=1.0, generation_length=384, n_samples=8, prompt_length=48)
