#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Pure Wideband Live RTL-SDR Capture Frontend with Push Button Recording Control
# Author: gr-playground agent
# Description: Live RTL-SDR RF pure wideband raw capture with interactive GUI frequency tuning and Push Button Start/Stop capture controls
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import osmosdr
import time
import sip



class rtlsdr_wideband_frontend(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Pure Wideband Live RTL-SDR Capture Frontend with Push Button Recording Control", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Pure Wideband Live RTL-SDR Capture Frontend with Push Button Recording Control")
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
        self.rf_gain = rf_gain = 49.6
        self.center_freq = center_freq = 100000000.0
        self.capture_enable = capture_enable = 0

        ##################################################
        # Blocks
        ##################################################

        self._rf_gain_range = qtgui.Range(0.0, 49.6, 1.0, 49.6, 200)
        self._rf_gain_win = qtgui.RangeWidget(self._rf_gain_range, self.set_rf_gain, "RF Tuner Gain (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._rf_gain_win)
        self._center_freq_range = qtgui.Range(88000000.0, 108000000.0, 100000.0, 100000000.0, 200)
        self._center_freq_win = qtgui.RangeWidget(self._center_freq_range, self.set_center_freq, "Center Frequency (Hz)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._center_freq_win)
        _capture_enable_push_button = Qt.QPushButton('Start / Stop Capture')
        _capture_enable_push_button = Qt.QPushButton('Start / Stop Capture')
        self._capture_enable_choices = {'Pressed': 1, 'Released': 0}
        _capture_enable_push_button.pressed.connect(lambda: self.set_capture_enable(self._capture_enable_choices['Pressed']))
        _capture_enable_push_button.released.connect(lambda: self.set_capture_enable(self._capture_enable_choices['Released']))
        self.top_layout.addWidget(_capture_enable_push_button)
        self.wideband_file_sink = blocks.file_sink(gr.sizeof_gr_complex*1, '/tmp/wideband_raw_capture.sigmf-data', False)
        self.wideband_file_sink.set_unbuffered(False)
        self.rtlsdr_source = osmosdr.source(
            args="numchan=" + str(1) + " " + 'rtl=0'
        )
        self.rtlsdr_source.set_time_unknown_pps(osmosdr.time_spec_t())
        self.rtlsdr_source.set_sample_rate(samp_rate)
        self.rtlsdr_source.set_center_freq(center_freq, 0)
        self.rtlsdr_source.set_freq_corr(0, 0)
        self.rtlsdr_source.set_dc_offset_mode(0, 0)
        self.rtlsdr_source.set_iq_balance_mode(0, 0)
        self.rtlsdr_source.set_gain_mode(False, 0)
        self.rtlsdr_source.set_gain(rf_gain, 0)
        self.rtlsdr_source.set_if_gain(20, 0)
        self.rtlsdr_source.set_bb_gain(20, 0)
        self.rtlsdr_source.set_antenna('', 0)
        self.rtlsdr_source.set_bandwidth(0, 0)
        self.qtgui_freq_sink = qtgui.freq_sink_c(
            2048, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            center_freq, #fc
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
        self.blocks_copy_0 = blocks.copy(gr.sizeof_gr_complex*1)
        self.blocks_copy_0.set_enabled(bool(capture_enable))


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_copy_0, 0), (self.wideband_file_sink, 0))
        self.connect((self.rtlsdr_source, 0), (self.blocks_copy_0, 0))
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
        self.rtlsdr_source.set_sample_rate(self.samp_rate)
        self.qtgui_freq_sink.set_frequency_range(self.center_freq, self.samp_rate)

    def get_rf_gain(self):
        return self.rf_gain

    def set_rf_gain(self, rf_gain):
        self.rf_gain = rf_gain
        self.rtlsdr_source.set_gain(self.rf_gain, 0)

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.rtlsdr_source.set_center_freq(self.center_freq, 0)
        self.qtgui_freq_sink.set_frequency_range(self.center_freq, self.samp_rate)

    def get_capture_enable(self):
        return self.capture_enable

    def set_capture_enable(self, capture_enable):
        self.capture_enable = capture_enable
        self.blocks_copy_0.set_enabled(bool(self.capture_enable))




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
