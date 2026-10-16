# Supplementary calibration, heads and trajectory evaluation review

Read calibrate_supplementary.py, train_supplementary_heads.py,
prepare_trajectory_sequences.py and evaluate_trajectories.py in full.
No implementation changes, provider execution or raw human data download occurred.

Supplementary calibration authenticates frozen states, raw adjacent calibration
rows and endpoint sidecars, checking case identity and actual calibration/user
roles at origin rows. Distinct users cannot cross buckets. Bounded per-user count
ratios expose independent-user/common-population/fixed-representation assumptions;
coverage for the human process is explicitly not established. Smoothed point
emissions, refined endpoint partitions and matched random controls have separate
integer certificates. These are one-step endpoint models, not general trajectory
or cognitive guarantees. The later supplementary-count integer correction remains
separate from historical source execution.

Outcome heads authenticate original encoder source/weights, training-only
preprocessing and fitting-cohort identities. Train and validation representations
are generated separately. Every endpoint regularization candidate is fit on
training rows and chosen by equal-user validation NLL with smaller-C tie breaking.
Unavailable training labels are explicit, and missing validation labels select
the lowest-C fallback. Final test rows are not read. Current corrected preprocessing
must not be substituted silently into historical reproduction.

Trajectory preparation reconstructs representatives in original physical-row
order, checks saved test-origin IDs and horizon order, and validates reconstructed
sequences against observed masks, positive counts and final outcomes. Users and
calendar roles constrain every observed future. This is recovery of already
recorded labels, not model prediction or causal inference. Private row-level
arrays stay on provider storage.

Trajectory evaluation uses frozen sampled original IDs, authenticated parent and
quotient arrays and saved origin-only forecasts. Controls match finite-model
capacity. Frozen neural origin probabilities form an explicitly labeled IID
comparator; no future context is fed into that comparator. Padded outcomes are
never scored beyond observed horizons. Unclipped sequence probabilities retain
infinite losses and explicit failure counts; comparisons require finite paired
user losses and minimum support. These sample-conditional findings cannot replace
the primary failed compression test or establish arbitrary future transfer.

Retained reconciliation verifies thirty head fits and 180 endpoint validation
selections, ten calibration reports with explicit coverage limitations, thirty-two
sequence bucket totals and all 450 trajectory metric/censoring cells. See
reports/preflight/supplementary-trajectory-controller-reconciliation.json.
No fresh private-array arithmetic, neural inference or certificate replay occurred.
Existing independent backward-likelihood audits remain separate evidence.

Calibration/evaluation launchers use lifetime locks and exclusive attempt markers;
head launchers have exclusive attempt markers but no coordinator lock. Sequence
preparation has a lock and retained handle, without an exclusive submission-attempt
marker. Their provider failure and concurrent cache paths were not exercised.
No tests were rerun for report-only changes. Strict final environment validation,
remaining source reviews and whole-release acceptance remain open.
