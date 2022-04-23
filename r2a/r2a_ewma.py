from r2a.ir2a import IR2A
from player.parser import *
import time
from statistics import mean
import numpy as np
import pandas as pd
from base.whiteboard import Whiteboard


class R2A_EWMA(IR2A):
    def __init__(self, id):
        IR2A.__init__(self, id)
        self.throughputs = []
        self.request_time = 0
        self.qi = []
        self.probs = []
        self.MINIMUM_BUFFER_SIZE = 15

    def handle_xml_request(self, msg):
        self.request_time = time.perf_counter()
        self.send_down(msg)

    def handle_xml_response(self, msg):

        parsed_mpd = parse_mpd(msg.get_payload())
        self.qi = parsed_mpd.get_qi()

        t = time.perf_counter() - self.request_time
        self.throughputs.append(msg.get_bit_length() / t)

        self.send_up(msg)

    def handle_segment_size_request(self, msg):
        self.request_time = time.perf_counter()

        # Get exponential weighted moving average for the throughput
        if self.throughputs:
            ewma_df = pd.DataFrame(self.throughputs).ewm(com=0.9999).mean()
            ewma = [x[0] for x in ewma_df.values][-1]

            # calculate probability
            avg = mean(self.throughputs)
            weight = 0
            for i in range(1, len(self.throughputs) + 1):
                weight = (i / len(self.throughputs)) * abs(
                    self.throughputs[i - 1] - avg
                )

            prob = avg / (avg + weight)
            self.probs.append(prob)

            # if there's buffer is half empty, full it with lower quality
            buf = self.whiteboard.get_playback_buffer_size()
            if buf and buf[-1][1] < self.MINIMUM_BUFFER_SIZE:
                ewma *= 0.5

            # selected desired quality
            selected_qi = min(self.qi)
            for i in self.qi:
                if ewma * prob >= i:
                    selected_qi = i
        else:
            selected_qi = min(self.qi)

        # request package
        msg.add_quality_id(selected_qi)
        self.send_down(msg)

    def handle_segment_size_response(self, msg):
        t = time.perf_counter() - self.request_time
        self.throughputs.append(msg.get_bit_length() / t)
        self.send_up(msg)

    def initialize(self):
        pass

    def finalization(self):
        pass
