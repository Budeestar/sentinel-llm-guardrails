# Injections hidden in long benign text

Model threshold 0.3670. Each scheme's threshold is calibrated on DEV-built long benign text to <= 1% FPR (never below the model threshold); numbers below are on TEST-built texts (400 of each kind, 5k/8k/12k chars). See the script docstring for the construction.

| Window (size/stride) | Threshold | Detection, own paragraph | Detection, inline | Long-benign FPR | ms/text |
| --- | ---: | --- | --- | --- | ---: |
| 4000/2000 (shipped) | 0.367 | 0.2% [0.0–1.4] | 0.0% [0.0–1.0] | 0.0% [0.0–1.0] | 22 |
| 1000/500 | 0.485 | 16.2% [13.0–20.2] | 16.0% [12.7–19.9] | 0.2% [0.0–1.4] | 27 |
| 600/300 | 0.746 | 21.2% [17.5–25.5] | 20.5% [16.8–24.7] | 0.5% [0.1–1.8] | 22 |
| 400/200 | 0.813 | 35.0% [30.5–39.8] | 35.2% [30.7–40.1] | 2.0% [1.0–3.9] | 23 |

For comparison, the same attack prompts on their own are detected at about 97% (results/classifier.md). A bag-of-features classifier averages one short attack with thousands of benign characters, so the signal is diluted; smaller windows recover some of it, but at a calibrated threshold most hidden attacks are still missed.
