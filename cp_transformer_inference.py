import numpy as np

from cp_transformer import RoFormerSymbolicTransformer, SOS_TOKEN, EOS_TOKEN
from preprocess_large_midi_dataset import preprocess_midi, DURATION_TEMPLATES
from settings import RWC_DATASET_PATH
import torch
import pretty_midi
import os

def decode_output(outputs, save_path, tempo=120.0):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = 60.0 / tempo / 4
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    for output in outputs:
        instrument_map = {}
        for time_step, data in enumerate(output):
            content = data.squeeze(0)
            start_time = time_step * time_step_length
            for i in range(0, len(content), 2):
                program = int(content[i].item())
                if program == EOS_TOKEN:
                    break
                if i + 1 >= len(content):
                    print('Incomplete note @', time_step, i)
                    break
                pitch_duration = int(content[i + 1].item()) - 128
                pitch = pitch_duration % 128
                duration = pitch_duration // 128
                if program < 0 or program >= 128:
                    print('Invalid program:', program, '@', time_step, i)
                    break
                if pitch < 0 or pitch >= 128:
                    print('Invalid pitch:', pitch, '@', time_step, i)
                    break
                if duration < 0 or duration >= len(DURATION_TEMPLATES):
                    print('Invalid duration:', duration, '@', time_step, i)
                    break
                end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time
                if program not in instrument_map:
                    if program == 127: # placeholder for drums
                        instrument_map[program] = pretty_midi.Instrument(0, is_drum=True)
                    else:
                        instrument_map[program] = pretty_midi.Instrument(program)
                    midi.instruments.append(instrument_map[program])
                instrument = instrument_map[program]
                instrument.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)


def decompress(model, byte_arr):
    x = torch.tensor(byte_arr).unsqueeze(0)
    x = x.cuda()
    return model.preprocess(x, pitch_shift=torch.zeros(1, dtype=torch.int8).cuda())

def continuation(model, midi_path, prompt_length=100, generation_length=384, temperature=1.0, n_samples=1):
    x = decompress(model, preprocess_midi(midi_path, 16)[0])
    print(x.shape)
    x = x[:, :prompt_length]
    decode_output([x[:, i, :] for i in range(x.shape[1])],
                  f'temp/{model.save_name}/{os.path.basename(midi_path)}_prompt.mid')
    with torch.no_grad():
        x = x.repeat(n_samples, 1, 1)
        output = model.global_sampling(x, temperature=temperature, max_seq_len=generation_length)
    for i in range(n_samples):
        output_i = [output[j][i:i + 1, :] for j in range(len(output))]
        decode_output(output_i, f'temp/{model.save_name}/{os.path.basename(midi_path)}_temp{temperature}_continuation_{i}.mid')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python cp_transformer_inference.py <model_path>')
        exit(1)
    model_path = sys.argv[1]
    model = RoFormerSymbolicTransformer.load_from_checkpoint(model_path, large=True)
    model.save_name = os.path.basename(model_path)
    model.cuda()
    model.eval()
    continuation(model, 'input/ashover1.mid', temperature=1.0, generation_length=384, n_samples=8, prompt_length=75)
    continuation(model, 'input/RM-P003.SMF_SYNC.MID', temperature=1.0, generation_length=384, n_samples=8)