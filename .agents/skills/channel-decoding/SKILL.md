---
name: channel-decoding
description: Receiver skill to perform Forward Error Correction (FEC) channel decoding (Viterbi K=7 R=1/2, Hamming(7,4), Repetition majority voting, Reed-Solomon RS(15,11)).
---

# Channel Decoding Skill 🛡️

Use this skill when demodulated bits require Forward Error Correction (FEC) decoding to correct channel transmission errors before passing data to L2 frame parsers.

## Instructions

1. **Execute Channel Decoding CLI Tool**:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:. .agents/skills/channel-decoding/scripts/decode_channel_code.py --input /tmp/demodulated_bits.bin --fec CONVOLUTIONAL_K7 --output /tmp/decoded_bits.bin
   ```
2. **Programmatic Usage**:
   ```python
   from gr_playground.dsp.channel_decoding import ChannelDecoder

   decoded_bits = ChannelDecoder.decode(demodulated_bits, fec_type="CONVOLUTIONAL_K7")
   ```
