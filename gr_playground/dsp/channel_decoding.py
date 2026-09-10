"""
Forward Error Correction (FEC) Channel Decoders
Implements:
- Repetition Decoder (Majority Voting)
- Hamming(7,4) Decoder (Syndrome Error Correction)
- Viterbi Decoder (Rate 1/2, Constraint Length K=7 Convolutional Code)
- Reed-Solomon RS(15,11) Decoder over GF(2^4)
"""

import numpy as np
from gnuradio import gr, blocks, fec

class ChannelDecoder:
    """Decoder for Forward Error Correction (FEC) schemes using GNU Radio flowgraphs."""

    @staticmethod
    def decode_repetition(bits: np.ndarray, rate: int = 3) -> np.ndarray:
        """Majority voting decoder for repetition codes using GNU Radio fec.repetition_decoder flowgraph."""
        bits_u8 = np.asarray(bits, dtype=np.uint8)
        n_blocks = len(bits_u8) // rate
        if n_blocks == 0:
            return np.array([], dtype=np.uint8)
            
        class RepetitionDecoderFlowgraph(gr.top_block):
            def __init__(self, in_bits, rep_rate, block_count):
                super().__init__("RepetitionDecoderFlowgraph")
                bits_trunc = in_bits[:block_count * rep_rate]
                # Convert 0/1 bits to soft decision floats: 0 -> -1.0, 1 -> +1.0
                soft_floats = (2.0 * bits_trunc - 1.0).astype(np.float32)
                
                dec_obj = fec.repetition_decoder_make(block_count, rep_rate, 0.5)
                self.src = blocks.vector_source_f(soft_floats.tolist(), False)
                self.dec = fec.decoder(dec_obj, gr.sizeof_float, gr.sizeof_char)
                self.sink = blocks.vector_sink_b()
                self.connect(self.src, self.dec, self.sink)

            def run_decode(self):
                self.run()
                return np.array(self.sink.data(), dtype=np.uint8)

        tb = RepetitionDecoderFlowgraph(bits_u8, rate, n_blocks)
        return tb.run_decode()

    @staticmethod
    def decode_hamming_7_4(bits: np.ndarray) -> np.ndarray:
        """
        Hamming(7,4) Systematic Code Decoder with single-bit error correction.
        Parity check equations:
        s0 = d0 ^ d1 ^ d2 ^ p0
        s1 = d1 ^ d2 ^ d3 ^ p1
        s2 = d0 ^ d1 ^ d3 ^ p2
        """
        bits = np.asarray(bits, dtype=np.uint8)
        n_blocks = len(bits) // 7
        blocks = bits[:n_blocks * 7].reshape(n_blocks, 7)
        
        decoded_bits = []
        for block in blocks:
            d0, d1, d2, d3, p0, p1, p2 = block[0], block[1], block[2], block[3], block[4], block[5], block[6]
            
            s0 = d0 ^ d1 ^ d2 ^ p0
            s1 = d1 ^ d2 ^ d3 ^ p1
            s2 = d0 ^ d1 ^ d3 ^ p2
            syndrome = (s2 << 2) | (s1 << 1) | s0
            
            # Syndrome mapping to error bit position (1-indexed)
            # 001 -> p0, 010 -> p1, 100 -> p2, 101 -> d0, 011 -> d1, 111 -> d2, 110 -> d3
            syndrome_to_pos = {
                0b101: 0, # d0
                0b111: 1, # d1
                0b011: 2, # d2
                0b110: 3, # d3
                0b001: 4, # p0
                0b010: 5, # p1
                0b100: 6, # p2
            }
            
            if syndrome in syndrome_to_pos:
                err_pos = syndrome_to_pos[syndrome]
                block[err_pos] ^= 1 # Correct bit error
                
            decoded_bits.extend([block[0], block[1], block[2], block[3]])
            
        return np.array(decoded_bits, dtype=np.uint8)

    @staticmethod
    def decode_convolutional_viterbi_k7_r12(bits: np.ndarray) -> np.ndarray:
        """
        Viterbi Decoder for Rate 1/2, K=7 Convolutional Code.
        Polynomials: G1 = 171 octal, G2 = 133 octal.
        States: 2^(K-1) = 64 states.
        """
        bits = np.asarray(bits, dtype=np.uint8)
        n_pairs = len(bits) // 2
        pairs = bits[:n_pairs * 2].reshape(n_pairs, 2)
        
        g1_poly = [1, 1, 1, 1, 0, 0, 1]
        g2_poly = [1, 0, 1, 1, 0, 1, 1]
        
        num_states = 64 # 2^6
        
        # Precompute expected outputs for state transitions
        outputs = np.zeros((num_states, 2, 2), dtype=np.uint8)
        next_states = np.zeros((num_states, 2), dtype=np.uint8)
        
        for state in range(num_states):
            state_bits = [(state >> (5 - i)) & 1 for i in range(6)]
            for in_bit in (0, 1):
                reg = [in_bit] + state_bits
                c1 = np.sum(np.array(reg) * g1_poly) % 2
                c2 = np.sum(np.array(reg) * g2_poly) % 2
                outputs[state, in_bit] = [c1, c2]
                next_state = ((state >> 1) | (in_bit << 5)) & 0x3F
                next_states[state, in_bit] = next_state

        path_metrics = np.full(num_states, 1e6, dtype=np.float32)
        path_metrics[0] = 0.0
        
        traceback = np.zeros((n_pairs, num_states), dtype=np.uint8)
        
        for t in range(n_pairs):
            rx_pair = pairs[t]
            new_metrics = np.full(num_states, 1e6, dtype=np.float32)
            
            for state in range(num_states):
                if path_metrics[state] >= 1e5:
                    continue
                for in_bit in (0, 1):
                    exp = outputs[state, in_bit]
                    n_state = next_states[state, in_bit]
                    branch_metric = np.sum(rx_pair != exp)
                    candidate = path_metrics[state] + branch_metric
                    if candidate < new_metrics[n_state]:
                        new_metrics[n_state] = candidate
                        traceback[t, n_state] = (state << 1) | in_bit
                        
            path_metrics = new_metrics

        best_state = np.argmin(path_metrics)
        decoded_reversed = []
        curr_state = best_state
        
        for t in range(n_pairs - 1, -1, -1):
            info = traceback[t, curr_state]
            prev_state = info >> 1
            in_bit = info & 1
            decoded_reversed.append(in_bit)
            curr_state = prev_state
            
        decoded_bits = np.array(decoded_reversed[::-1], dtype=np.uint8)
        if len(decoded_bits) > 6:
            decoded_bits = decoded_bits[:-6]
            
        return decoded_bits

    @staticmethod
    def decode_reed_solomon_15_11(coded_bytes: bytes) -> bytes:
        """
        Reed-Solomon RS(15,11) over GF(2^4) Decoder with error correction.
        """
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

        def gf_add(a, b):
            return a ^ b

        # Unpack bytes into 4-bit nibble symbols
        nibbles = []
        for b in coded_bytes:
            nibbles.append((b >> 4) & 0x0F)
            nibbles.append(b & 0x0F)

        n_blocks = len(nibbles) // 15
        decoded_nibbles = []

        for b_idx in range(n_blocks):
            block = nibbles[b_idx * 15 : (b_idx + 1) * 15]
            
            # Evaluate syndromes S_1, S_2, S_3, S_4 where S_i = r(alpha^i)
            syndromes = [0] * 4
            for i in range(4):
                alpha_i = exp_table[i + 1]
                val = 0
                for j in range(15):
                    val = gf_add(block[j], gf_mul(val, alpha_i))
                syndromes[i] = val

            # Check if all syndromes are 0 (no errors)
            if any(s != 0 for s in syndromes):
                # Simple single-symbol or double-symbol error correction attempt
                # For single symbol error at position pos with magnitude mag:
                # S1 = mag * alpha^(15-1-pos)
                # S2 = mag * alpha^(2*(15-1-pos))
                s1, s2 = syndromes[0], syndromes[1]
                if s1 != 0 and s2 != 0:
                    ratio = gf_mul(s2, exp_table[15 - log_table[s1]])
                    if ratio in log_table:
                        alpha_power = log_table[ratio]
                        pos = 15 - 1 - alpha_power
                        if 0 <= pos < 15:
                            mag = gf_mul(s1, exp_table[15 - alpha_power])
                            block[pos] ^= mag

            # Extract 11 systematic message nibbles
            decoded_nibbles.extend(block[:11])

        # Repack nibbles to bytes
        out_bytes = bytearray()
        for i in range(0, len(decoded_nibbles), 2):
            high = decoded_nibbles[i]
            low = decoded_nibbles[i+1] if (i+1) < len(decoded_nibbles) else 0
            out_bytes.append((high << 4) | low)
            
        return bytes(out_bytes)

    @classmethod
    def decode(cls, data: np.ndarray | bytes, fec_type: str = "CONVOLUTIONAL_K7", **kwargs) -> np.ndarray | bytes:
        """Unified channel decoder dispatcher."""
        ft = fec_type.upper()
        if ft in ["REPETITION", "REP3"]:
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)) if isinstance(data, bytes) else data
            return cls.decode_repetition(bits, rate=kwargs.get("rate", 3))
        elif ft in ["HAMMING", "HAMMING_7_4", "HAMMING74"]:
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)) if isinstance(data, bytes) else data
            return cls.decode_hamming_7_4(bits)
        elif ft in ["CONVOLUTIONAL", "CONVOLUTIONAL_K7", "VITERBI"]:
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)) if isinstance(data, bytes) else data
            return cls.decode_convolutional_viterbi_k7_r12(bits)
        elif ft in ["REED_SOLOMON", "RS1511", "RS"]:
            coded_bytes = data if isinstance(data, bytes) else np.packbits(data).tobytes()
            return cls.decode_reed_solomon_15_11(coded_bytes)
        else:
            raise ValueError(f"Unsupported FEC channel coding type: {fec_type}")
