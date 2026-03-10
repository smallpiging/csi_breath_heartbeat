#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: yck
# GNU Radio version: 3.10.12.0

import os
import sys
import logging as log

def get_state_directory() -> str:
    oldpath = os.path.expanduser("~/.grc_gnuradio")
    try:
        from gnuradio.gr import paths
        newpath = paths.persistent()
        if os.path.exists(newpath):
            return newpath
        if os.path.exists(oldpath):
            log.warning(f"Found persistent state path '{newpath}', but file does not exist. " +
                     f"Old default persistent state path '{oldpath}' exists; using that. " +
                     "Please consider moving state to new location.")
            return oldpath
        # Default to the correct path if both are configured.
        # neither old, nor new path exist: create new path, return that
        os.makedirs(newpath, exist_ok=True)
        return newpath
    except (ImportError, NameError):
        log.warning("Could not retrieve GNU Radio persistent state directory from GNU Radio. " +
                 "Trying defaults.")
        xdgstate = os.getenv("XDG_STATE_HOME", os.path.expanduser("~/.local/state"))
        xdgcand = os.path.join(xdgstate, "gnuradio")
        if os.path.exists(xdgcand):
            return xdgcand
        if os.path.exists(oldpath):
            log.warning(f"Using legacy state path '{oldpath}'. Please consider moving state " +
                     f"files to '{xdgcand}'.")
            return oldpath
        # neither old, nor new path exist: create new path, return that
        os.makedirs(xdgcand, exist_ok=True)
        return xdgcand

sys.path.append(os.environ.get('GRC_HIER_PATH', get_state_directory()))

from gnuradio import blocks
import pmt
from gnuradio import uhd
import time
from wifi_phy_hier import wifi_phy_hier  # grc-generated hier_block
import foo
import ieee802_11
import threading
import wifi_loopback_real_no_pyqt_epy_block_0 as epy_block_0  # embedded python block
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation




class wifi_loopback_real_no_pyqt(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.tx_gain = tx_gain = 0.75
        self.samp_rate = samp_rate = 20e6
        self.rx_gain = rx_gain = 0.75
        self.pdu_length = pdu_length = 2
        self.out_buf_size = out_buf_size = 96000
        self.lo_offset = lo_offset = 0
        self.interval = interval = 5
        self.freq = freq = 5920000000
        self.encoding = encoding = 0
        self.chan_est = chan_est = 0

        ##################################################
        # Blocks
        ##################################################

        self.wifi_phy_hier_0 = wifi_phy_hier(
            bandwidth=10e6,
            chan_est=ieee802_11.Equalizer(chan_est),
            encoding=int(ieee802_11.Encoding(encoding)),
            frequency=5.9e6,
            sensitivity=0.56,
        )
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(('', "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_source_0.set_center_freq(uhd.tune_request(freq, rf_freq = freq - lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_normalized_gain(rx_gain, 0)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(('', "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            'packet_len',
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(freq, rf_freq = freq - lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_normalized_gain(tx_gain, 0)
        self.ieee802_11_parse_mac_0 = ieee802_11.parse_mac(False, False)
        self.ieee802_11_mac_0 = ieee802_11.mac([0x23, 0x23, 0x23, 0x23, 0x23, 0x23], [0x42, 0x42, 0x42, 0x42, 0x42, 0x42], [0xff, 0xff, 0xff, 0xff, 0xff, 0xff])
        self.ieee802_11_extract_csi_0 = ieee802_11.extract_csi()
        self.foo_packet_pad2_0 = foo.packet_pad2(False, False, 0.001, 500, 0)
        self.foo_packet_pad2_0.set_min_output_buffer((out_buf_size * 10))
        self.epy_block_0 = epy_block_0.blk()
        self.blocks_message_strobe_0 = blocks.message_strobe(pmt.intern("".join("x" for i in range(pdu_length))), interval)
        self.blocks_file_sink_1 = blocks.file_sink(gr.sizeof_gr_complex*52, '/home/yck/桌面/毕业设计/python/数据集制作终/data', False)
        self.blocks_file_sink_1.set_unbuffered(True)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0, 'strobe'), (self.ieee802_11_mac_0, 'app in'))
        self.msg_connect((self.ieee802_11_mac_0, 'phy out'), (self.wifi_phy_hier_0, 'mac_in'))
        self.msg_connect((self.ieee802_11_parse_mac_0, 'out'), (self.ieee802_11_extract_csi_0, 'pdu in'))
        self.msg_connect((self.wifi_phy_hier_0, 'mac_out'), (self.ieee802_11_parse_mac_0, 'in'))
        self.connect((self.foo_packet_pad2_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.ieee802_11_extract_csi_0, 0), (self.blocks_file_sink_1, 0))
        self.connect((self.ieee802_11_extract_csi_0, 0), (self.epy_block_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.wifi_phy_hier_0, 0))
        self.connect((self.wifi_phy_hier_0, 0), (self.foo_packet_pad2_0, 0))


    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.uhd_usrp_sink_0.set_normalized_gain(self.tx_gain, 0)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.uhd_usrp_source_0.set_normalized_gain(self.rx_gain, 0)

    def get_pdu_length(self):
        return self.pdu_length

    def set_pdu_length(self, pdu_length):
        self.pdu_length = pdu_length
        self.blocks_message_strobe_0.set_msg(pmt.intern("".join("x" for i in range(self.pdu_length))))

    def get_out_buf_size(self):
        return self.out_buf_size

    def set_out_buf_size(self, out_buf_size):
        self.out_buf_size = out_buf_size

    def get_lo_offset(self):
        return self.lo_offset

    def set_lo_offset(self, lo_offset):
        self.lo_offset = lo_offset
        self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(self.freq, rf_freq = self.freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_source_0.set_center_freq(uhd.tune_request(self.freq, rf_freq = self.freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)

    def get_interval(self):
        return self.interval

    def set_interval(self, interval):
        self.interval = interval
        self.blocks_message_strobe_0.set_period(self.interval)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(self.freq, rf_freq = self.freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_source_0.set_center_freq(uhd.tune_request(self.freq, rf_freq = self.freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)

    def get_encoding(self):
        return self.encoding

    def set_encoding(self, encoding):
        self.encoding = encoding
        self.wifi_phy_hier_0.set_encoding(int(ieee802_11.Encoding(self.encoding)))

    def get_chan_est(self):
        return self.chan_est

    def set_chan_est(self, chan_est):
        self.chan_est = chan_est
        self.wifi_phy_hier_0.set_chan_est(ieee802_11.Equalizer(self.chan_est))




def main(top_block_cls=wifi_loopback_real_no_pyqt, options=None):
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        gr.logger("realtime").warn("Error: failed to enable real-time scheduling.")
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
