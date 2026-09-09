"""
Layer-2 Framing Identification & Message Signal Extraction DSP Module
Automatically identifies L2 framing formats from demodulated bit/byte streams:
- HDLC
- COBS
- AX.25
- CCSDS
- IEEE 802.15.4
- Generic Length-Prefix
Validates CRCs / checksums, parses frame metadata, and extracts message signal payload bytes.
Includes bit-shift alignment search (0-7 bits) and phase inversion tolerance.
"""

import struct
import numpy as np
from typing import Dict, Any, Union

from gr_playground.simulator.framing import crc16_ccitt, crc32_ieee, cobs_decode

class L2FramingIdentifier:
    """Receiver module to identify L2 framing scheme and extract message payload."""

    @staticmethod
    def _bits_to_bytes(bits: np.ndarray) -> bytes:
        """Convert uint8 bit array to bytes."""
        bits = np.asarray(bits, dtype=np.uint8)
        n_bytes = len(bits) // 8
        if n_bytes == 0:
            return b""
        bits_trunc = bits[:n_bytes * 8]
        return np.packbits(bits_trunc).tobytes()

    @classmethod
    def _scan_bytes(cls, raw_bytes: bytes) -> Dict[str, Any]:
        """Scan a single byte array for all supported L2 framing formats."""
        if not raw_bytes:
            return None

        # 1. Check CCSDS (Attached Sync Marker: 0x1ACFFC1D)
        asm_idx = raw_bytes.find(b"\x1A\xCF\xFC\x1D")
        if asm_idx != -1 and len(raw_bytes) >= asm_idx + 12:
            frame_data = raw_bytes[asm_idx + 4 :]
            if len(frame_data) >= 8:
                first_word, seq_count, length_field = struct.unpack(">HHH", frame_data[:6])
                vcid = first_word & 0x3F
                mcid = (first_word >> 6) & 0xFF
                payload_len = length_field
                
                if len(frame_data) >= 6 + payload_len + 2:
                    payload = frame_data[6 : 6 + payload_len]
                    rx_crc = struct.unpack(">H", frame_data[6 + payload_len : 8 + payload_len])[0]
                    calc_crc = crc16_ccitt(frame_data[: 6 + payload_len])
                    is_valid = (rx_crc == calc_crc)
                    
                    if is_valid:
                        return {
                            "framing_type": "CCSDS",
                            "is_valid_crc": True,
                            "frame_header": {"mcid": mcid, "vcid": vcid, "seq_count": seq_count, "length": payload_len},
                            "extracted_message": payload,
                            "extracted_message_str": payload.decode("utf-8", errors="ignore"),
                            "confidence": 1.0
                        }

        # 2. Check Generic Length Prefix (Sync word: 0x352E6B4D)
        sync_idx = raw_bytes.find(b"\x35\x2E\x6B\x4D")
        if sync_idx != -1 and len(raw_bytes) >= sync_idx + 12:
            frame_data = raw_bytes[sync_idx :]
            if len(frame_data) >= 8:
                sync, length_field, seq_num = struct.unpack(">4sHH", frame_data[:8])
                if len(frame_data) >= 8 + length_field + 4:
                    payload = frame_data[8 : 8 + length_field]
                    rx_crc = struct.unpack(">I", frame_data[8 + length_field : 12 + length_field])[0]
                    calc_crc = crc32_ieee(frame_data[: 8 + length_field])
                    is_valid = (rx_crc == calc_crc)
                    
                    if is_valid:
                        return {
                            "framing_type": "GENERIC_LENGTH_PREFIX",
                            "is_valid_crc": True,
                            "frame_header": {"seq_num": seq_num, "length": length_field},
                            "extracted_message": payload,
                            "extracted_message_str": payload.decode("utf-8", errors="ignore"),
                            "confidence": 1.0
                        }

        # 3. Check IEEE 802.15.4 (Preamble: 0x00000000A7)
        ieee_idx = raw_bytes.find(b"\x00\x00\x00\x00\xA7")
        if ieee_idx != -1 and len(raw_bytes) >= ieee_idx + 15:
            frame_len = raw_bytes[ieee_idx + 5]
            psdu = raw_bytes[ieee_idx + 6 : ieee_idx + 6 + frame_len]
            if len(psdu) >= 11:
                fcf, seq_num, dest_pan, dest_addr, src_addr = struct.unpack("<HBHHH", psdu[:9])
                payload = psdu[9:-2]
                rx_fcs = struct.unpack("<H", psdu[-2:])[0]
                calc_fcs = crc16_ccitt(psdu[:-2])
                is_valid = (rx_fcs == calc_fcs)
                
                if is_valid:
                    return {
                        "framing_type": "IEEE_802_15_4",
                        "is_valid_crc": True,
                        "frame_header": {"fcf": hex(fcf), "seq_num": seq_num, "dest_pan": hex(dest_pan), "dest_addr": hex(dest_addr), "src_addr": hex(src_addr)},
                        "extracted_message": payload,
                        "extracted_message_str": payload.decode("utf-8", errors="ignore"),
                        "confidence": 1.0
                    }

        # 4. Check HDLC / AX.25 (Delimiter Flag: 0x7E)
        flag_positions = [i for i, b in enumerate(raw_bytes) if b == 0x7E]
        if len(flag_positions) >= 2:
            for f_idx in range(len(flag_positions) - 1):
                start_flag = flag_positions[f_idx]
                end_flag = flag_positions[f_idx + 1]
                if end_flag > start_flag + 4:
                    frame_body = raw_bytes[start_flag + 1 : end_flag]
                    
                    unescaped = bytearray()
                    i = 0
                    while i < len(frame_body):
                        if frame_body[i] == 0x7D and i + 1 < len(frame_body):
                            unescaped.append(frame_body[i+1] ^ 0x20)
                            i += 2
                        else:
                            unescaped.append(frame_body[i])
                            i += 1
                            
                    unescaped_bytes = bytes(unescaped)
                    if len(unescaped_bytes) >= 4:
                        rx_crc = struct.unpack("<H", unescaped_bytes[-2:])[0]
                        calc_crc = crc16_ccitt(unescaped_bytes[:-2])
                        is_valid = (rx_crc == calc_crc)
                        
                        if is_valid:
                            if len(unescaped_bytes) >= 16:
                                dest_call_bytes = bytes([b >> 1 for b in unescaped_bytes[:6]])
                                if dest_call_bytes.isalnum() or b"CQ" in dest_call_bytes:
                                    ctrl, pid = unescaped_bytes[14], unescaped_bytes[15]
                                    payload = unescaped_bytes[16:-2]
                                    return {
                                        "framing_type": "AX.25",
                                        "is_valid_crc": True,
                                        "frame_header": {"dest_call": dest_call_bytes.decode("ascii", errors="ignore"), "ctrl": ctrl, "pid": pid},
                                        "extracted_message": payload,
                                        "extracted_message_str": payload.decode("utf-8", errors="ignore"),
                                        "confidence": 1.0
                                    }
                                    
                            address, control = unescaped_bytes[0], unescaped_bytes[1]
                            payload = unescaped_bytes[2:-2]
                            return {
                                "framing_type": "HDLC",
                                "is_valid_crc": True,
                                "frame_header": {"address": address, "control": control},
                                "extracted_message": payload,
                                "extracted_message_str": payload.decode("utf-8", errors="ignore"),
                                "confidence": 1.0
                            }

        # 5. Check COBS (0x00 delimiter)
        cobs_positions = [i for i, b in enumerate(raw_bytes) if b == 0x00]
        if cobs_positions:
            start_pos = 0
            for c_pos in cobs_positions:
                if c_pos > start_pos + 3:
                    cobs_block = raw_bytes[start_pos:c_pos]
                    try:
                        decoded_cobs = cobs_decode(cobs_block)
                        if len(decoded_cobs) >= 3:
                            payload = decoded_cobs[:-2]
                            rx_crc = struct.unpack("<H", decoded_cobs[-2:])[0]
                            calc_crc = crc16_ccitt(payload)
                            if rx_crc == calc_crc:
                                return {
                                    "framing_type": "COBS",
                                    "is_valid_crc": True,
                                    "frame_header": {"cobs_len": len(cobs_block)},
                                    "extracted_message": payload,
                                    "extracted_message_str": payload.decode("utf-8", errors="ignore"),
                                    "confidence": 1.0
                                }
                    except Exception:
                        pass
                start_pos = c_pos + 1

        return None

    @classmethod
    def identify_and_extract(cls, data: Union[np.ndarray, bytes]) -> Dict[str, Any]:
        """
        Inspect demodulated bit/byte stream across bit-shifts (0..7) and polarities, detect L2 framing, validate CRC, and extract payload.
        """
        if isinstance(data, bytes):
            # Direct byte search
            res = cls._scan_bytes(data)
            if res:
                return res
            # Convert bytes to bit array for shift/inversion search
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        else:
            bits = np.asarray(data, dtype=np.uint8)

        if len(bits) == 0:
            return {
                "framing_type": "UNKNOWN",
                "is_valid_crc": False,
                "frame_header": {},
                "extracted_message": b"",
                "extracted_message_str": "",
                "confidence": 0.0
            }

        # Try normal and inverted bit polarities, across 8 bit shifts
        for invert in (False, True):
            test_bits = (1 - bits) if invert else bits
            for shift in range(8):
                if len(test_bits) <= shift + 16:
                    continue
                shifted_bits = test_bits[shift:]
                raw_bytes = cls._bits_to_bytes(shifted_bits)
                
                res = cls._scan_bytes(raw_bytes)
                if res:
                    return res

        # Fallback if no valid CRC framing pattern found
        raw_fallback = cls._bits_to_bytes(bits)
        return {
            "framing_type": "UNKNOWN_UNFRAMED",
            "is_valid_crc": False,
            "frame_header": {},
            "extracted_message": raw_fallback,
            "extracted_message_str": raw_fallback.decode("utf-8", errors="ignore"),
            "confidence": 0.1
        }
