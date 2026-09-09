"""
L2 Framing Generators / Encoders
Implements standard Layer-2 framing schemes:
- HDLC (High-Level Data Link Control)
- COBS (Consistent Overhead Byte Stuffing)
- AX.25 (Amateur Packet Radio Frame)
- CCSDS (Space Data Link Transfer Frame)
- IEEE 802.15.4 MAC Frame
- Generic Sync + Length Prefix Frame
Includes standard CRC-16 and CRC-32 calculation.
"""

import struct
import numpy as np

def crc16_ccitt(data: bytes, poly=0x1021, init=0xFFFF) -> int:
    """Calculate CRC-16 CCITT checksum."""
    crc = init
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

def crc32_ieee(data: bytes) -> int:
    """Calculate CRC-32 IEEE 802.3 checksum."""
    import zlib
    return zlib.crc32(data) & 0xFFFFFFFF

def cobs_encode(data: bytes) -> bytes:
    """COBS (Consistent Overhead Byte Stuffing) encoder."""
    dest = bytearray()
    code_idx = 0
    code = 1
    dest.append(0)
    for b in data:
        if b != 0:
            dest.append(b)
            code += 1
            if code == 0xFF:
                dest[code_idx] = code
                code_idx = len(dest)
                dest.append(0)
                code = 1
        else:
            dest[code_idx] = code
            code_idx = len(dest)
            dest.append(0)
            code = 1
    dest[code_idx] = code
    return bytes(dest)

def cobs_decode(data: bytes) -> bytes:
    """COBS (Consistent Overhead Byte Stuffing) decoder."""
    dest = bytearray()
    idx = 0
    length = len(data)
    while idx < length:
        code = data[idx]
        if code == 0:
            break
        idx += 1
        for _ in range(1, code):
            if idx >= length:
                break
            dest.append(data[idx])
            idx += 1
        if code < 0xFF and idx < length and data[idx] != 0:
            dest.append(0)
    return bytes(dest)

class L2FrameEncoder:
    """Encoder for Layer-2 Framing protocols."""

    @staticmethod
    def encode_hdlc(payload: bytes, address: int = 0xFF, control: int = 0x03) -> bytes:
        """
        Encode payload into HDLC frame.
        Format: Flag (0x7E) | Address (1B) | Control (1B) | Payload | CRC-16 (2B) | Flag (0x7E)
        """
        header = bytes([address & 0xFF, control & 0xFF])
        body = header + payload
        crc = crc16_ccitt(body)
        crc_bytes = struct.pack("<H", crc)
        frame_content = body + crc_bytes
        
        # HDLC Flag delimiting with byte escaping for zero-bit transparency emulation
        escaped = bytearray()
        for b in frame_content:
            if b in (0x7E, 0x7D):
                escaped.extend([0x7D, b ^ 0x20])
            else:
                escaped.append(b)
                
        return bytes([0x7E]) + bytes(escaped) + bytes([0x7E])

    @staticmethod
    def encode_cobs(payload: bytes) -> bytes:
        """
        Encode payload into COBS frame.
        Format: COBS_Encode( Payload | CRC-16 (2B) ) | 0x00 Delimiter
        """
        crc = crc16_ccitt(payload)
        data_with_crc = payload + struct.pack("<H", crc)
        encoded = cobs_encode(data_with_crc)
        return encoded + bytes([0x00])

    @staticmethod
    def encode_ax25(payload: bytes, dest_callsign: str = "CQ    ", src_callsign: str = "N0CALL", control: int = 0x03, pid: int = 0xF0) -> bytes:
        """
        Encode payload into AX.25 Amateur Radio frame.
        Format: Flag (0x7E) | Dest Callsign (7B) | Src Callsign (7B) | Control (1B) | PID (1B) | Payload | CRC-16 | Flag (0x7E)
        """
        def format_callsign(call: str, ssid: int = 0, last: bool = False) -> bytes:
            call_padded = call.upper().ljust(6)[:6]
            res = bytearray([ord(c) << 1 for c in call_padded])
            ssid_byte = (0x60 | ((ssid & 0x0F) << 1) | (1 if last else 0)) & 0xFF
            res.append(ssid_byte)
            return bytes(res)

        dest_b = format_callsign(dest_callsign, 0, False)
        src_b = format_callsign(src_callsign, 0, True)
        
        header = dest_b + src_b + bytes([control & 0xFF, pid & 0xFF])
        body = header + payload
        crc = crc16_ccitt(body)
        crc_bytes = struct.pack("<H", crc)
        frame_content = body + crc_bytes
        
        return bytes([0x7E]) + frame_content + bytes([0x7E])

    @staticmethod
    def encode_ccsds(payload: bytes, vcid: int = 1, mcid: int = 0, seq_count: int = 1) -> bytes:
        """
        Encode payload into CCSDS Space Transfer Frame.
        Format: ASM (0x1ACFFC1D, 4B) | Version/MCID/VCID (2B) | Seq (2B) | Length (2B) | Payload | CRC-16 (2B)
        """
        asm = bytes([0x1A, 0xCF, 0xFC, 0x1D])
        first_word = ((0 & 0x03) << 14) | ((mcid & 0xFF) << 6) | (vcid & 0x3F)
        length_field = len(payload) & 0xFFFF
        header = struct.pack(">HHH", first_word, seq_count & 0x3FFF, length_field)
        
        body = header + payload
        crc = crc16_ccitt(body)
        crc_bytes = struct.pack(">H", crc)
        return asm + body + crc_bytes

    @staticmethod
    def encode_ieee802154(payload: bytes, seq_num: int = 1, dest_pan: int = 0xFFFF, dest_addr: int = 0xFFFF, src_addr: int = 0x1234) -> bytes:
        """
        Encode payload into IEEE 802.15.4 MAC Frame.
        Format: Preamble (4x 0x00) | SFD (0xA7) | Frame Length (1B) | FCF (2B) | Seq (1B) | Dest PAN (2B) | Dest Addr (2B) | Src Addr (2B) | Payload | FCS (CRC-16)
        """
        preamble = bytes([0x00, 0x00, 0x00, 0x00, 0xA7])
        fcf = 0x8841 # Standard data frame, 16-bit addressing
        mac_header = struct.pack("<HBHHH", fcf, seq_num & 0xFF, dest_pan, dest_addr, src_addr)
        psdu_payload = mac_header + payload
        crc = crc16_ccitt(psdu_payload)
        fcs = struct.pack("<H", crc)
        
        full_psdu = psdu_payload + fcs
        frame_len = len(full_psdu) & 0xFF
        return preamble + bytes([frame_len]) + full_psdu

    @staticmethod
    def encode_generic_length_prefix(payload: bytes, sync_word: bytes = b"\x35\x2E\x6B\x4D", seq_num: int = 1) -> bytes:
        """
        Encode payload into Generic Length-Prefixed Frame.
        Format: Sync Word (4B) | Length (2B) | Seq (2B) | Payload | CRC-32 (4B)
        """
        length = len(payload)
        header = sync_word + struct.pack(">HH", length, seq_num & 0xFFFF)
        body = header + payload
        crc = crc32_ieee(body)
        return body + struct.pack(">I", crc)

    @classmethod
    def encode(cls, payload: bytes, framing_type: str = "HDLC", **kwargs) -> bytes:
        """Unified L2 framing encode dispatcher."""
        ft = framing_type.upper()
        if ft == "HDLC":
            return cls.encode_hdlc(payload, **kwargs)
        elif ft == "COBS":
            return cls.encode_cobs(payload)
        elif ft in ["AX25", "AX.25"]:
            return cls.encode_ax25(payload, **kwargs)
        elif ft == "CCSDS":
            return cls.encode_ccsds(payload, **kwargs)
        elif ft in ["IEEE802154", "IEEE_802_15_4", "802.15.4"]:
            return cls.encode_ieee802154(payload, **kwargs)
        elif ft in ["GENERIC", "GENERIC_LENGTH_PREFIX", "LENGTH_PREFIX"]:
            return cls.encode_generic_length_prefix(payload, **kwargs)
        else:
            raise ValueError(f"Unsupported L2 framing type: {framing_type}")
