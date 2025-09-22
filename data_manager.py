import os


def find_all_midi_files(root_dir):
    midi_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mid') or filename.lower().endswith('.midi'):
                midi_files.append(os.path.abspath(os.path.join(dirpath, filename)))
    return midi_files
