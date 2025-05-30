import pickle

def write_to_file(file_path, content):
  with open(file_path, 'wb') as f:
    pickle.dump(content, f)

def read_from_file(file_path):
  with open(file_path, 'rb') as f:
    return pickle.load(f)
