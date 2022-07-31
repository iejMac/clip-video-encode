

class NumpyEncoder:
    """class that watches directory for set of numpy arrays of videos to encode using CLIP."""
    def __init__(self, data_dir, dest_dir, vids):
        """
            
        Input:
            data_dir: directory to watch for np files
            dest_dir:  where to save embeddings to
            vids: list of numpy array names to watch for (completes when all fnmaes have been seen).
        """
        self.data_dir = data_dir
        self.dest_dir = dest_dir
        self.vids = vids


    def start(self):
        print(self.data_dir)
        print(self.dest_dir)
        print(len(self.vids))
