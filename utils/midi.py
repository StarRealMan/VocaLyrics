from mido import MidiFile


QUANT_STEP = 0.25
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


def pitch_to_name(pitch):
    if pitch < 0 or pitch > 127:
        return None
    name = NOTE_NAMES[pitch % 12]
    octave = pitch // 12 - 1
    return f"{name}{octave}"

def parse_midi(midi_path: str):
    mid = MidiFile(midi_path)
    tpq = mid.ticks_per_beat  # ticks per quarter note

    time_signatures = []
    tempos = []
    key = None

    current_ticks_meta = 0
    if len(mid.tracks) > 0:
        for msg in mid.tracks[0]:
            current_ticks_meta += msg.time

            if msg.type == "time_signature":
                time_signatures.append((msg.numerator, msg.denominator))

            elif msg.type == "set_tempo":
                bpm = 60000000 / msg.tempo if msg.tempo else 120.0
                tempos.append(bpm)

            elif msg.type == "key_signature" and key is None:
                key = msg.key

    # 默认值
    if not time_signatures:
        time_signatures.append((4, 4))
    if not tempos:
        tempos.append(120.0)
    if key is None:
        key = "unknown"

    ts_num, ts_den = time_signatures[0]
    time_signature_str = f"{ts_num}/{ts_den}"
    bpm = tempos[0]

    selected_notes = []

    for mtrack in mid.tracks:
        current_ticks = 0
        # active_notes[(channel, pitch)] = (start_ticks, velocity)
        active_notes = {}
        notes = []

        for msg in mtrack:
            current_ticks += msg.time
            time_beats = current_ticks / tpq
            time_beats = round(time_beats / QUANT_STEP) * QUANT_STEP

            # note_on（velocity > 0） -> 开始一个音
            if msg.type == "note_on" and msg.velocity > 0:
                channel = getattr(msg, "channel", 0)
                pitch = msg.note
                velocity = msg.velocity
                key_note = (channel, pitch)
                active_notes[key_note] = (current_ticks, velocity)

            # note_off 或 note_on(velocity == 0) -> 结束一个音
            elif msg.type in ("note_off", "note_on") and msg.velocity == 0:
                channel = getattr(msg, "channel", 0)
                pitch = msg.note
                key_note = (channel, pitch)

                if key_note in active_notes:
                    start_ticks, velocity = active_notes.pop(key_note)
                    start_beats = start_ticks / tpq
                    start_beats = round(start_beats / QUANT_STEP) * QUANT_STEP
                    end_beats = time_beats

                    # 避免 0 或负时长
                    if end_beats > start_beats:
                        notes.append({
                            "start_beats": start_beats,
                            "end_beats": end_beats,
                            "name": pitch_to_name(pitch),
                            "velocity": velocity,
                        })

        # 如果这个轨道有音符，就选它，并停止继续扫后面的轨道
        if notes:
            selected_notes = notes
            break

    song_dict = {
        "meta": {
            "time_signature": time_signature_str,  # 如 "4/4"
            "bpm": bpm,                            # 如 120
            "key": key,                            # 如 "C" / "Am" / "unknown"
        },
        "notes": selected_notes,                   # list[dict]
    }

    return song_dict
