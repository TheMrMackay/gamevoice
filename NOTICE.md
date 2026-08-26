# Third-party components

GameVoice itself is MIT licensed (see `LICENSE`). It relies on the following,
each under its own terms. Nothing here is redistributed in this repository — the
installer fetches each component from its own source at install time.

## Speech synthesis

| Component | Licence | Notes |
|---|---|---|
| [Piper](https://github.com/OHF-Voice/piper1-gpl) (`piper-tts`) | MIT | Runs entirely offline |
| [onnxruntime](https://onnxruntime.ai/) | MIT | CPU inference |

## Voice models

Downloaded from the [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices)
collection. **Each voice carries its own licence** and they are not uniform.

| Voice | Corpus | Licence |
|---|---|---|
| `en_US-libritts_r-medium` | LibriTTS-R | CC BY 4.0 |
| `en_US-ryan-high` | Ryan Speech | CC BY 4.0 |
| `en_US-hfc_female-medium` | HiFi-CAPTAIN | CC BY-NC-SA 4.0 |

> **If you stream, record or publish audio produced by GameVoice, check the
> licence of the voice you used first.** A non-commercial voice is fine for
> personal listening and not for monetised content. `voices.json` in the
> collection lists the licence for every voice.

## Interface and capture

| Component | Licence |
|---|---|
| [PySide6](https://doc.qt.io/qtforpython/) (Qt for Python) | LGPL v3 |
| [mss](https://github.com/BoboTiG/python-mss) | MIT |
| [Pillow](https://python-pillow.org/) | MIT-CMU |
| [NumPy](https://numpy.org/) | BSD-3-Clause |
| [sounddevice](https://python-sounddevice.readthedocs.io/) | MIT |
| [pywin32](https://github.com/mhammond/pywin32) | PSF-2.0 |
| PyWinRT (`winrt-*`) | MIT |

Text recognition uses the OCR engine built into Windows, through the Windows
Runtime. No model is downloaded and nothing leaves the machine.

## What GameVoice does with what it reads

Text recognised from the screen is converted to speech locally and played back.
It is not stored, transmitted or sent to any service. Reading a game's on-screen
text aloud for your own use is the intended purpose; redistributing a recording
of copyrighted dialogue is not something this licence grants you.
