---
name: l2-framing-identification
description: Receiver skill to identify Layer-2 framing schemes (HDLC, COBS, AX.25, CCSDS, IEEE 802.15.4, Generic Length-Prefix), parse headers, check CRCs, and extract message signals.
---

# Layer-2 Framing Identification Skill 🏷️

Use this skill when you need to inspect demodulated bit or byte payloads, identify which Layer-2 protocol framing was applied (HDLC, COBS, AX.25, CCSDS, IEEE 802.15.4, or Generic Length-Prefixed), validate CRC/FCS checksum integrity, and extract the underlying message payload.

## Instructions

1. **Execute Framing Identification CLI Tool**:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:. .agents/skills/l2-framing-identification/scripts/identify_l2_framing.py --input /tmp/demodulated_bits.bin
   ```
2. **Programmatic Usage**:
   ```python
   from gr_playground.dsp.l2_framing_id import L2FramingIdentifier

   results = L2FramingIdentifier.identify_and_extract(demodulated_bytes_or_bits)
   print("Framing Type:", results["framing_type"])
   print("CRC Valid:", results["is_valid_crc"])
   print("Extracted Message:", results["extracted_message_str"])
   ```
