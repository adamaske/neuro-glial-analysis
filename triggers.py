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
fnirs_info = StreamInfo('ExperimentMarkerStream', 'Markers', 1, 0, 'string') 
#MISSING : What stream to output to? WHat does Aurora / EEG Listen to
eeg_info = StreamInfo('EEG-MarkresStream', 'Markers', 1, 0, 'string')

# next make an outlet
fnirs_outlet = StreamOutlet(fnirs_info)
eeg_outlet = StreamOutlet(eeg_info)

#              0,     1,       2,        3      4
markers = ["brest", "srest", "right", "left", "end"]
durations = [40, 20, 10, 10, 0]

start = input("press any key to start....")
def push_marker_and_wait(index):
    fnirs_outlet.push_sample(markers[index])
    eeg_outlet.push_sample(markers[index])
    time.sleep(durations[index])

push_marker_and_wait(0) #BIG REST
push_marker_and_wait(2) #RIGHT
push_marker_and_wait(1) #SMALL REST
push_marker_and_wait(3) #LEFT

push_marker_and_wait(0)
push_marker_and_wait(2)
push_marker_and_wait(1)
push_marker_and_wait(3)

push_marker_and_wait(0)
push_marker_and_wait(2)
push_marker_and_wait(1)
push_marker_and_wait(3)

push_marker_and_wait(0) # BIG REST
push_marker_and_wait(4) #end of program
exit()

counter = 0
while counter < :
    # pick a sample to send an wait for a bit
    outlet.push_sample([random.choice(markernames)])
    time.sleep(random.random()*3)