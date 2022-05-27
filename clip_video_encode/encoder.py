import time

def encode_videos(frames, done_loading, batch_size):
  # while not done_loading[0]:
  
  for i in range(10):
    frames = frames[1:]
    print(len(frames))
    time.sleep(0.01)


