"""Example program to demonstrate how to send string-valued markers into LSL."""

import random
import time

from pylsl import StreamInfo, StreamOutlet

# first create a new stream info (here we set the name to MyMarkerStream,
# the content-type to Markers, 1 channel, irregular sampling rate,
# and string-valued data) The last value would be the locally unique
# identifier for the stream as far as available, e.g.
# program-scriptname-subjectnumber (you could also omit it but interrupted
# connections wouldn't auto-recover). The important part is that the
# content-type is set to 'Markers', because then other programs will know how
#  to interpret the content
fnirs_info = StreamInfo(name='Trigger', type='Markers', channel_count=1, channel_format='int32', source_id='Example') 
#MISSING : What stream to output to? WHat does Aurora / EEG Listen t
#eeg_info = StreamInfo('EEG-MarkresStream', 'Markers', 1, 0, 'string')

# next make an outlet
fnirs_outlet = StreamOutlet(fnirs_info)
#eeg_outlet = StreamOutlet(eeg_info)

#              0,           1,          2,       3      4      
markers = ["BIG REST", "SMALL REST", "RIGHT", "LEFT", "END" ]
durations = [40, 20, 10, 10]

order = [0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0]

def push_marker(index):
    fnirs_outlet.push_sample([index])
    #eeg_outlet.push_sample(markers[index])

start = input("press any key to start experiment....")

start_time = time.time()

idx = 0
current_block_idx = 0

running = True
while idx < len(order):
    block_started = time.time()  # When did the current block start
    current_block_idx = order[idx] #find the index in markers list

    print(f"Processing block {markers[current_block_idx]}")
    print(f"SUBJECT EXECUTE : {markers[current_block_idx]}")
    
    push_marker(order[idx])
    while time.time() - block_started < durations[current_block_idx]:
        elapsed_time = time.time() - block_started
        remaining_time = durations[current_block_idx] - elapsed_time
        
        # Print the remaining time (rounded to 2 decimal places)
        print(f"Remaining time for block {current_block_idx}: {remaining_time:.2f} seconds", end='\r')
        time.sleep(0.1)  # Sleep a bit to prevent excessive printing (optional)
        
    idx += 1  # Move to the next block
    print(f"Block {current_block_idx} completed, moving to next.")

push_marker(4) #finished push \end\
print(f"EXPERIMENT COMPLETE")
exit()
