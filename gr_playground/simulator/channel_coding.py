"""
Forward Error Correction (FEC) Channel Encoders
Implements:
- Repetition Code (Rate 1/3, Rate 1/5)
- Hamming(7,4) Code
- Convolutional Code (Rate 1/2, Constraint Length K=7)
- Reed-Solomon RS(15,11) Code over GF(2^4)
"""

import numpy as np
import struct

class ChannelEncoder:
    """Encoder for Forward Error Correction (FEC) schemes."""

    @staticmethod
    def encode_repetition(bits: np.ndarray, rate: int = 3) -> np.ndarray:
        """Repeat each bit `rate` times."""
        bits = np.asarray(bits, dtype=np.uint8)
        return np.repeat(bits, rate)

    @staticmethod
    def encode_hamming_7_4(bits: np.ndarray) -> np.ndarray:
        """
        Hamming(7,4) Systematic Code Encoder.
        Generator Matrix G = [I4 | P]
        p0 = d0 ^ d1 ^ d2
        p1 = d1 ^ d2 ^ d3
        p2 = d0 ^ d1 ^ d3
        """
        bits = np.asarray(bits, dtype=np.uint8)
        # Pad to multiple of 4
        remainder = len(bits) % 4
        if remainder != 0:
            pad = 4 - remainder
            bits = np.pad(bits, (0, pad), mode="constant", constant_values=0)
            
        nibbles = bits.reshape(-1, 4)
        encoded_blocks = []
        for nibble in nibbles:
            d0, d1, d2, d3 = nibble[0], nibble[1], nibble[2], nibble[3]
            p0 = d0 ^ d1 ^ d2
            p1 = d1 ^ d2 ^ d3
            p2 = d0 ^ d1 ^ d3
            code_word = [d0, d1, d2, d3, p0, p1, p2]
            encoded_blocks.append(code_word)
            
        return np.array(encoded_blocks, dtype=np.uint8).flatten()

    @staticmethod
    def encode_convolutional_k7_r12(bits: np.ndarray) -> np.ndarray:
        """
        Rate 1/2, K=7 Convolutional Encoder.
        Polynomials: G1 = 171 (octal) = 0b1111001 (121 decimal)
                     G2 = 133 (octal) = 0b1011011 (91 decimal)
        """
        bits = np.asarray(bits, dtype=np.uint8)
        g1_poly = [1, 1, 1, 1, 0, 0, 1]
        g2_poly = [1, 0, 1, 1, 0, 1, 1]
        
        shift_reg = np.zeros(7, dtype=np.uint8)
        encoded_bits = []
        
        # Flush with 6 zero bits to reset trellis at the end
        padded_bits = np.pad(bits, (0, 6), mode="constant", constant_values=0)
        
        for bit in padded_bits:
            shift_reg = np.roll(shift_reg, 1)
            shift_reg[0] = bit & 1
            
            c1 = np.sum(shift_reg * g1_poly) % 2
            c2 = np.sum(shift_reg * g2_poly) % 2
            encoded_bits.extend([c1, c2])
            
        return np.array(encoded_bits, dtype=np.uint8)

    @staticmethod
    def encode_reed_solomon_15_11(data_bytes: bytes) -> bytes:
        """
        Simplified Reed-Solomon RS(15,11) over GF(2^4) Encoder.
        Maps 11 4-bit nibbles (5.5 bytes) into 15 4-bit nibbles (7.5 bytes), adding 4 parity nibbles.
        Primitive polynomial p(x) = x^4 + x + 1 (19 decimal).
        """
        # GF(2^4) arithmetic table setup
        exp_table = [0] * 32
        log_table = [0] * 16
        x = 1
        for i in range(15):
            exp_table[i] = x
            exp_table[i + 15] = x
            log_table[x] = i
            x <<= 1
            if x & 0x10:
                x ^= 0x13
        log_table[0] = 0

        def gf_mul(a, b):
            if a == 0 or b == 0:
                return 0
            return exp_table[log_table[a] + log_table[b]]

        # Generator polynomial for 4 parity symbols (t=2 error correction):
        # g(x) = (x + alpha^1)(x + alpha^2)(x + alpha^3)(x + alpha^4)
        # g(x) = x^4 + 15*x^3 + 3*x^2 + 1*x + 12 (coefficients in GF(2^4))
        g = [1, 15, 3, 1, 12]

        # Convert input bytes to 4-bit nibble symbols
        nibbles = []
        for b in data_bytes:
            nibbles.append((b >> 4) & 0x0F)
            nibbles.append(b & 0x0F)

        # Pad nibbles to multiple of 11
        rem = len(nibbles) % 11
        if rem != 0:
            nibbles.extend([0] * (11 - rem))

        coded_nibbles = []
        for i in range(0, len(nibbles), 11):
            block = nibbles[i:i+11]
            # Systematic RS encoding: multiply msg by x^4 and compute remainder mod g(x)
            reg = [0] * 4
            for symbol in block:
                feedback = symbol ^ reg[0]
                reg[0] = reg[1] ^ gf_mul(feedback, g[1])
                reg[1] = reg[2] ^ gf_mul(feedback, g[2])
                reg[2] = reg[3] ^ gf_mul(feedback, g[3])
                reg[3] = gf_mul(feedback, g[4])
            
            coded_block = block + reg
            coded_nibbles.extend(coded_block)

        # Repack nibbles back to bytes
        out_bytes = bytearray()
        for i in range(0, len(coded_nibbles), 2):
            high = coded_nibbles[i]
            low = coded_nibbles[i+1] if (i+1) < len(coded_nibbles) else 0
            out_bytes.append((high << 4) | low)
            
        return bytes(out_bytes)

    @classmethod
    def encode(cls, data: np.ndarray | bytes, fec_type: str = "CONVOLUTIONAL_K7", **kwargs) -> np.ndarray | bytes:
        """Unified channel encoder dispatcher."""
        ft = fec_type.upper()
        if ft in ["REPETITION", "REP3"]:
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)) if isinstance(data, bytes) else data
            return cls.encode_repetition(bits, rate=kwargs.get("rate", 3))
        elif ft in ["HAMMING", "HAMMING_7_4", "HAMMING74"]:
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)) if isinstance(data, bytes) else data
            return cls.encode_hamming_7_4(bits)
        elif ft in ["CONVOLUTIONAL", "CONVOLUTIONAL_K7", "VITERBI"]:
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)) if isinstance(data, bytes) else data
            return cls.encode_convolutional_k7_r12(bits)
        elif ft in ["REED_SOLOMON", "RS1511", "RS"]:
            data_bytes = data if isinstance(data, bytes) else np.packbits(data).tobytes()
            return cls.encode_reed_solomon_15_11(data_bytes)
        else:
            raise ValueError(f"Unsupported FEC channel coding type: {fec_type}")
