from r2a.ir2a import IR2A
from player.parser import *
import time
from statistics import mean
import numpy as np
from base.whiteboard import Whiteboard


class R2A_EWMA(IR2A):
    def __init__(self, id):
        IR2A.__init__(self, id)
        self.throughputs = []
        self.request_time = 0
        self.qi = []

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
        window_size = 10
        sum_proportion = 0.9999
        alpha = 1 - np.exp(np.log(1 - sum_proportion) / window_size)

        # Get exponential weighted moving average for the throughput
        ewma = ewma_vectorized_safe(self.throughputs, alpha)[-1]

        print()
        # considering the current throughput, what is the necessary quality to download to keep the buffer at minimum in a size where it covers possible new pauses?
        history = self.whiteboard.get_playback_history()
        pauses = self.whiteboard.get_playback_pauses()
        print(f"{pauses=}")
        if history and pauses:
            history_pause = max([x for x, y in history if x >= pauses[-1][0]])
            if history_pause - pauses[-1][0] >= window_size * 6:
                buf_pause = ewma_vectorized_safe(
                    [y for x, y in self.whiteboard.get_playback_pauses()], alpha
                )
                pause = buf_pause[-1] if len(buf_pause) else 5
                # decrease needed new blocks to buffer from current blocks in buffer
                # TODO: bufs_need =
                print(f"{buf_pause=}")
                print(f"{pause=}")
                print(f"ewma b: {ewma}")
                ewma = (ewma * 2 + ewma / pause) / (2 + pause)
        ewma *= 0.8
        print(f"ewma a: {ewma}")
        print(f"{self.throughputs=}")

        selected_qi = self.qi[0]
        for i in self.qi:
            if ewma > i:
                selected_qi = i
        if all(y == 0 for x, y in history):
            selected_qi = self.qi[0]
        print(f"{selected_qi=}")
        print()

        # buf = self.whiteboard.get_playback_buffer_size()
        # buf_ = buf if len(buf) <= 5 else buf[-5::]
        # pauses = self.whiteboard.get_playback_pauses()
        # pauses_ = pauses if len(pauses) <= 5 else pauses[-5::]
        # history = self.whiteboard.get_playback_history()
        # history_ = history if len(history) <= 5 else history[-5::]

        # print(f'{buf=}')
        # print(f'{pauses=}')
        # print(f'{history=}')
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


def ewma_vectorized_safe(data, alpha, row_size=None, dtype=None, order="C", out=None):
    """
    Reshapes data before calculating EWMA, then iterates once over the rows
    to calculate the offset without precision issues
    :param data: Input data, will be flattened.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param row_size: int, optional
        The row size to use in the computation. High row sizes need higher precision,
        low values will impact performance. The optimal value depends on the
        platform and the alpha being used. Higher alpha values require lower
        row size. Default depends on dtype.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    :return: The flattened result.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    print(f"{row_size=}")
    row_size = (
        int(row_size)
        if (row_size is not None and int(row_size) > 0)
        else get_max_row_size(alpha, dtype)
    )

    if data.size <= row_size:
        # The normal function can handle this input, use that
        return ewma_vectorized(data, alpha, dtype=dtype, order=order, out=out)

    if data.ndim > 1:
        # flatten input
        data = np.reshape(data, -1, order=order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    assert row_size != 0, "Divisor can't be zero."

    row_n = int(data.size // row_size)  # the number of rows to use
    trailing_n = int(data.size % row_size)  # the amount of data leftover
    first_offset = data[0]

    if trailing_n > 0:
        # set temporary results to slice view of out parameter
        out_main_view = np.reshape(out[:-trailing_n], (row_n, row_size))
        data_main_view = np.reshape(data[:-trailing_n], (row_n, row_size))
    else:
        out_main_view = out
        data_main_view = data

    # get all the scaled cumulative sums with 0 offset
    ewma_vectorized_2d(
        data_main_view,
        alpha,
        axis=1,
        offset=0,
        dtype=dtype,
        order="C",
        out=out_main_view,
    )

    scaling_factors = (1 - alpha) ** np.arange(1, row_size + 1)
    last_scaling_factor = scaling_factors[-1]

    # create offset array
    offsets = np.empty(out_main_view.shape[0], dtype=dtype)
    offsets[0] = first_offset
    # iteratively calculate offset for each row
    for i in range(1, out_main_view.shape[0]):
        offsets[i] = offsets[i - 1] * last_scaling_factor + out_main_view[i - 1, -1]

    # add the offsets to the result
    out_main_view += offsets[:, np.newaxis] * scaling_factors[np.newaxis, :]

    if trailing_n > 0:
        # process trailing data in the 2nd slice of the out parameter
        ewma_vectorized(
            data[-trailing_n:],
            alpha,
            offset=out_main_view[-1, -1],
            dtype=dtype,
            order="C",
            out=out[-trailing_n:],
        )
    return out


def get_max_row_size(alpha, dtype=float):
    assert 0.0 <= alpha < 1.0
    # This will return the maximum row size possible on
    # your platform for the given dtype. I can find no impact on accuracy
    # at this value on my machine.
    # Might not be the optimal value for speed, which is hard to predict
    # due to numpy's optimizations
    # Use np.finfo(dtype).eps if you  are worried about accuracy
    # and want to be extra safe.
    epsilon = np.finfo(dtype).tiny
    # If this produces an OverflowError, make epsilon larger
    return int(np.log(epsilon) / np.log(1 - alpha)) + 1


def ewma_vectorized(data, alpha, offset=None, dtype=None, order="C", out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(
        1.0 - alpha, np.arange(data.size + 1, dtype=dtype), dtype=dtype
    )
    # create cumulative sum array
    np.multiply(
        data, (alpha * scaling_factors[-2]) / scaling_factors[:-1], dtype=dtype, out=out
    )
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out
