import numpy as np
import xf_midi
import os
from joblib import Parallel, delayed
import torch
import shutil
import json
import argparse


DURATION_TEMPLATES = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096])



def preprocess_midi(midi_path, max_polyphony, beat_div=4, ins_ids='all'):
    '''
    input: midi file path
    output: (1) a tensor representing the music content of shape (seq_len, sub_seq_len) where seq_len is number of seconds and sub_seq_len is 48 (max_polyphony * [program, pitch, duration])
            (2) a tensor representing the range of pitch in the whole song
    '''

    # print(midi_path)
    try:
        midi = xf_midi.XFMidi(midi_path, constant_tempo=60.0 / beat_div)
    except:
        return None
    # print(midi)
    midi_end_time = int(midi.get_end_time())
    print("midi_end_time:",midi_end_time)
    # print(midi_end_time)
    if midi_end_time <= 0:
        return None
    if not isinstance(ins_ids, list):
        ins_ids = [ins_ids]
    duration_boundaries = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2
    # print(duration_boundaries)
    min_pitch = 127
    max_pitch = 0
    result_rolls = []
    # print(ins_ids)
    for ins_id in ins_ids:
        has_any_note = False
        rolls = np.full((midi_end_time, max_polyphony, 3), dtype=np.uint8, fill_value=255)
        polyphony_counts = np.zeros(midi_end_time, dtype=np.uint8)
        for i, ins in enumerate(midi.instruments):
            # print(i, ins)
            program = ins.program
            if ins.is_drum:
                program = 127
            for note in ins.notes:
                start_time = int(round(note.start))
                end_time = int(round(note.end))
                if start_time >= 0 and end_time < midi_end_time and polyphony_counts[start_time] < max_polyphony:
                    if ins.is_drum:
                        duration = 0
                    else:
                        duration = np.searchsorted(duration_boundaries, end_time - start_time) #为了做四舍五入的quantize
                        min_pitch = min(min_pitch, note.pitch)
                        max_pitch = max(max_pitch, note.pitch)
                    add_note = False
                    if ins_id == 'all':
                        add_note = True
                    elif isinstance(ins_id, int):
                        raise NotImplementedError
                    elif isinstance(ins_id, str):
                        if '-' in ins_id:
                            task, num = ins_id.split('-')
                            num = int(num)
                            if task == 'track':
                                add_note = i == num
                            elif task == 'upto':
                                add_note = i <= num
                            elif task == 'from':
                                add_note = i >= num
                            elif task == 'notrack':
                                add_note = i != num
                            else:
                                raise NotImplementedError
                        elif ins_id == 'drum':
                            add_note = ins.is_drum
                        elif ins_id == 'nondrum':
                            add_note = not ins.is_drum
                        elif ins_id == 'empty':
                            add_note = False
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                    if add_note:
                        has_any_note = True
                        rolls[start_time, polyphony_counts[start_time]] = [program, note.pitch, duration]
                        # [program, pitch, duration]
                        polyphony_counts[start_time] += 1
        if not has_any_note and ins_id != 'empty':
            return None  # invalid midi file
        for i in range(midi_end_time):
            # Sort notes by ins first, then by pitch, then by duration
            rolls[i, :polyphony_counts[i]] = rolls[i, :polyphony_counts[i]][np.lexsort((rolls[i, :polyphony_counts[i], 2], rolls[i, :polyphony_counts[i], 1], rolls[i, :polyphony_counts[i], 0]))]
            if polyphony_counts[i] < max_polyphony:
                rolls[i, polyphony_counts[i], 0] = 254  # EOS token
        result_rolls.append(rolls)
    result_rolls = np.concatenate(result_rolls, axis=1)
    # print(result_rolls)
    # Get song-level pitch shift range
    pitch_shift_max = 127 - max_pitch
    pitch_shift_min = -min_pitch
    print(midi_path, ": final", torch.tensor(result_rolls.reshape(midi_end_time, -1)).shape, torch.tensor([pitch_shift_min, pitch_shift_max], dtype=torch.int8).shape)
    return torch.tensor(result_rolls.reshape(midi_end_time, -1)), torch.tensor([pitch_shift_min, pitch_shift_max], dtype=torch.int8)


def create_npy_dataset_from_midi(folder, max_polyphony, dataset_name, ins_ids='all', scan_subfolders=True, max_idx=None):
    '''
    combine the data from all midi files in a folder to a single tensor
    '''

    with open("output_rwc.json", "r", encoding="utf-8") as f:
        midi_files = json.load(f)
    if max_idx is not None:
        midi_files = midi_files[:max_idx]
    midi_files = [midi_file.replace('FixChannel10', 'rwc_chord').replace('SMF_SYNC', 'MIDI.CHORD') for midi_file in midi_files]
    print(f'Processing {len(midi_files)} files')
    results = Parallel(n_jobs=-1, verbose=10)(delayed(preprocess_midi)(midi_file, max_polyphony, ins_ids=ins_ids) for midi_file in midi_files)
    # Filter out None results
    results = [result for result in results if result is not None]
    results_data = [result[0] for result in results]
    results_shift = [result[1] for result in results]
    # np.save(f'data/{dataset_name}.npy', np.concatenate(results, axis=0))
    torch.save(torch.cat(results_data, dim=0), f'data/{dataset_name}.pt')
    torch.save(torch.cat(results_shift, dim=0), f'data/{dataset_name}.pitch_shift_range.pt')
    lengths = [len(data) for data in results_data]
    # np.save(f'data/{dataset_name}.length.npy', np.array(lengths))
    torch.save(torch.tensor(lengths), f'data/{dataset_name}.length.pt')


def parse_args():
    parser = argparse.ArgumentParser(description="Preprcoess MIDI folder")
    parser.add_argument("--midi_folder", '-f', type=str, required=True, help="midi folder")
    parser.add_argument("--dataset_name", '-n', type=str, required=True, help="naming output")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    create_npy_dataste_from_midi(args.midi_folder, 16, args.dataset_name)

    # why the tensors for single midi files are concat into one big tensor?
    # so that the tensor can be easily divided into equal length chunks later 
