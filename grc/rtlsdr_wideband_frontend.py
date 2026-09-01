#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: RTL-SDR Wideband Capture & Channelizer Frontend
# Author: gr-playground agent
# Description: Live RTL-SDR RF capture and wideband file channelizer exporting to SigMF
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import soapy
import sip



class rtlsdr_wideband_frontend(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "RTL-SDR Wideband Capture & Channelizer Frontend", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("RTL-SDR Wideband Capture & Channelizer Frontend")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "rtlsdr_wideband_frontend")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 2400000
        self.channel_cutoff = channel_cutoff = 60000.0
        self.lpf_taps = lpf_taps = firdes.low_pass(1.0, samp_rate, channel_cutoff,channel_cutoff * 0.2, window.WIN_HAMMING, 6.76)
        self.channel_freq_offset = channel_freq_offset = 200000.0

        ##################################################
        # Blocks
        ##################################################

        self._channel_freq_offset_range = qtgui.Range(-1000000.0, 1000000.0, 1000.0, 200000.0, 200)
        self._channel_freq_offset_win = qtgui.RangeWidget(self._channel_freq_offset_range, self.set_channel_freq_offset, "Channel Offset (Hz)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._channel_freq_offset_win)
        self.agc = analog.agc2_cc((1e-3), (1e-2), 1.0, 1.0, 65536)
        self.sigmf_file_sink = blocks.file_sink(gr.sizeof_gr_complex*1, '/tmp/extracted_channel.sigmf-data', False)
        self.sigmf_file_sink.set_unbuffered(False)
        self.rtlsdr_source = None
        dev = 'driver=rtlsdr'
        stream_args = 'bufflen=16384'
        tune_args = ['']
        settings = ['']

        def _set_rtlsdr_source_gain_mode(channel, agc):
            self.rtlsdr_source.set_gain_mode(channel, agc)
            if not agc:
                  self.rtlsdr_source.set_gain(channel, self._rtlsdr_source_gain_value)
        self.set_rtlsdr_source_gain_mode = _set_rtlsdr_source_gain_mode

        def _set_rtlsdr_source_gain(channel, name, gain):
            self._rtlsdr_source_gain_value = gain
            if not self.rtlsdr_source.get_gain_mode(channel):
                self.rtlsdr_source.set_gain(channel, gain)
        self.set_rtlsdr_source_gain = _set_rtlsdr_source_gain

        def _set_rtlsdr_source_bias(bias):
            if 'biastee' in self._rtlsdr_source_setting_keys:
                self.rtlsdr_source.write_setting('biastee', bias)
        self.set_rtlsdr_source_bias = _set_rtlsdr_source_bias

        self.rtlsdr_source = soapy.source(dev, "fc32", 1, '',
                                  stream_args, tune_args, settings)

        self._rtlsdr_source_setting_keys = [a.key for a in self.rtlsdr_source.get_setting_info()]

        self.rtlsdr_source.set_sample_rate(0, samp_rate)
        self.rtlsdr_source.set_frequency(0, 100.0e6)
        self.rtlsdr_source.set_frequency_correction(0, 0)
        self.set_rtlsdr_source_bias(bool(False))
        self._rtlsdr_source_gain_value = 20
        self.set_rtlsdr_source_gain_mode(0, bool(False))
        self.set_rtlsdr_source_gain(0, 'TUNER', 20)
        self.qtgui_freq_sink = qtgui.freq_sink_c(
            2048, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            100.0e6, #fc
            samp_rate, #bw
            "Wideband Spectrum (RTL-SDR)", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink.set_update_time(0.10)
        self.qtgui_freq_sink.set_y_axis((-140), 10)
        self.qtgui_freq_sink.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink.enable_autoscale(False)
        self.qtgui_freq_sink.enable_grid(True)
        self.qtgui_freq_sink.set_fft_average(0.2)
        self.qtgui_freq_sink.enable_axis_labels(True)
        self.qtgui_freq_sink.enable_control_panel(False)
        self.qtgui_freq_sink.set_fft_window_normalized(False)



        labels = ['Wideband Spectrum', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink.set_line_label(i, labels[i])
            self.qtgui_freq_sink.set_line_width(i, widths[i])
            self.qtgui_freq_sink.set_line_color(i, colors[i])
            self.qtgui_freq_sink.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_win = sip.wrapinstance(self.qtgui_freq_sink.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_win)
        self.freq_xlating_filter = filter.freq_xlating_fir_filter_ccc(10, lpf_taps, channel_freq_offset, samp_rate)
        self._channel_cutoff_range = qtgui.Range(5000.0, 250000.0, 2500.0, 60000.0, 200)
        self._channel_cutoff_win = qtgui.RangeWidget(self._channel_cutoff_range, self.set_channel_cutoff, "Channel Cutoff (Hz)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._channel_cutoff_win)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.agc, 0), (self.sigmf_file_sink, 0))
        self.connect((self.freq_xlating_filter, 0), (self.agc, 0))
        self.connect((self.rtlsdr_source, 0), (self.freq_xlating_filter, 0))
        self.connect((self.rtlsdr_source, 0), (self.qtgui_freq_sink, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "rtlsdr_wideband_frontend")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_lpf_taps(firdes.low_pass(1.0, self.samp_rate, self.channel_cutoff, self.channel_cutoff * 0.2, window.WIN_HAMMING, 6.76))
        self.rtlsdr_source.set_sample_rate(0, self.samp_rate)
        self.qtgui_freq_sink.set_frequency_range(100.0e6, self.samp_rate)

    def get_channel_cutoff(self):
        return self.channel_cutoff

    def set_channel_cutoff(self, channel_cutoff):
        self.channel_cutoff = channel_cutoff
        self.set_lpf_taps(firdes.low_pass(1.0, self.samp_rate, self.channel_cutoff, self.channel_cutoff * 0.2, window.WIN_HAMMING, 6.76))

    def get_lpf_taps(self):
        return self.lpf_taps

    def set_lpf_taps(self, lpf_taps):
        self.lpf_taps = lpf_taps
        self.freq_xlating_filter.set_taps(self.lpf_taps)

    def get_channel_freq_offset(self):
        return self.channel_freq_offset

    def set_channel_freq_offset(self, channel_freq_offset):
        self.channel_freq_offset = channel_freq_offset
        self.freq_xlating_filter.set_center_freq(self.channel_freq_offset)




def main(top_block_cls=rtlsdr_wideband_frontend, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
