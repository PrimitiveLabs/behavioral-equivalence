# Theory comparison and remaining review obligations

Checked October 15, 2026. This is a source-based internal review, not independent proof review or a comprehensive novelty determination.

## Finite-horizon trace guarantees

[Bian and Abate, extended version v3](https://arxiv.org/html/1701.04547v3), Definition 4 and Theorems 3.1, 4.1 through 4.3, directly establish the existing connection between approximate bisimulation and bounded trace-event probabilities. Theorem 4.1 gives the same 1-(1-epsilon)^H form used here. Their relation-based definition permits matching successors through a relation; our implementation certifies a specific partition against its actual quotient rows. The local coupling proof verifies that operational condition rather than treating every approximate-equivalence definition as interchangeable. Our event language is bounded observable label paths; this argument does not automatically preserve nested branching-time probability tests. Trace agreement does not imply bisimulation. The finite-horizon formula is established prior work and cannot be presented as our novel theorem.

## Interval abstractions and representative choice

[Zacchia Lun, Wheatley, D'Innocenzo and Abate, v1](https://arxiv.org/html/1804.08554v1), Sections 1 through 3, construct interval-based abstractions of finite labeled Markov chains and consider virtual representatives beyond existing concrete states. Thus selecting a synthetic representative or combining abstraction with interval checking is not sufficient novelty. Our arithmetic-mean quotient is an implementable choice, not a demonstrated minimax representative. The uncertainty hull also permits hidden-state mixtures and per-step rectangular choices, which can be conservative relative to a stationary concrete kernel. A quantitative, like-for-like comparison with this paper's abstraction remains outstanding; no superiority claim is justified by the current benchmark alone.

## Predictive histories and causal states

[Shalizi and Crutchfield, author publication page](https://csc.ucdavis.edu/~cmg/compmech/pubs/cmppss.htm), identifies causal-state representations and predictive minimality as established computational-mechanics results. Our finite-history encoder, limited property suite and empirical partition do not establish the all-future predictive equivalence or optimality of an epsilon-machine. Compression of a learned model also does not identify unique psychological states. The full [arXiv v2 paper](https://arxiv.org/pdf/cond-mat/9907176) is now inspected. Sections III A, V and VII B assume a discrete-valued, discrete-time stationary process and compare predictive partitions of complete pasts. Its minimality and almost-sure uniqueness statements concern prescient representations, not arbitrary finite-history neural clusters. Our current-context inputs and transport across calendar changes do not establish those premises.

## Controlled predictive representations

[Littman, Sutton and Singh, author-hosted paper](https://web.eecs.umich.edu/~baveja/Papers/psr.pdf) and [official proceedings entry](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html) describe states through action-conditioned future-observation predictions. The paper's author listing includes Singh; the proceedings landing page displays only the other two authors. Our fixed-policy Markov experiment does not establish action-conditioned predictive sufficiency, learn the linear system dimension, or replace controlled-system identification. Training future-outcome representations is therefore an established direction, while our empirical certificate-aware integration remains a hypothesis to evaluate. The full [proceedings PDF](https://papers.nips.cc/paper_files/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf) is now inspected. Section 1 defines sufficiency for every action-observation test; Theorem 1 constructs a linear PSR no larger than a finite POMDP representation. Our finite fixed-policy event suite neither establishes sufficiency for all controlled tests nor measures that linear dimension. The denominator in the PSR update also emphasizes that conditioning requires supported observation probabilities.

## Consequences for this program

The defensible candidate contribution is an auditable implementation and experimental study: learned partitions, exact-normalized numerical certificates, operational uncertainty accounting, matched refinement controls, and independent human evaluation. The last component remains incomplete. Synthetic positive results and early human compression failures must both appear in the final claims.

Independent review must still check: the concrete-to-abstract coupling construction and event scope; exact numeric normalization and enclosure induction; uncertainty-family quantifiers and sampling assumptions; conditional-event denominators; finite-property versus general trace preservation; and the complete literature on foundational probabilistic bisimulation, sequential confidence procedures, and active experimental design. This document narrows unsupported claims but does not close those obligations.

## Classical probabilistic equivalence

[Larsen and Skou, 1991](https://vbn.aau.dk/en/publications/bisimulation-through-probabilistic-testing/) is a foundational probabilistic-testing and bisimulation reference. The institutional bibliographic record verifies authorship and publication; its abstract does not replace a full theorem audit. We do not claim the stable-partition definition or testing interpretation as new.

## Sequential uncertainty

[Howard, Ramdas, McAuliffe and Sekhon](https://arxiv.org/html/1810.08240v4), Sections 2 and 6, develop confidence sequences with simultaneous-in-time validity, including arbitrary stopping under stated exponential-supermartingale conditions. Our fixed-design Hoeffding row experiment is not a confidence sequence. Its guarantee cannot be reused for adaptively stopped sampling merely by checking it repeatedly. The active synthetic search queries a specified oracle; no adaptively sampled human confidence sequence has been evaluated. Learning representations or sharing calendar shocks also does not establish the required martingale premise. Sequential uncertainty remains future work rather than a feature claimed by this release.

## Review conclusion

The inspected foundations rule out novelty claims for finite-horizon approximate-bisimulation bounds, predictive history equivalence, action-observation predictive representations, interval abstractions and time-uniform inference. The present contribution is the implemented, audited experiment and its measured tradeoffs. No superiority over all prior abstraction constructions, psychological minimality, or new general soundness theorem is asserted. External peer review remains unperformed; it is a publication-quality limitation, not evidence of failure or a substitute for the analytical proof.
