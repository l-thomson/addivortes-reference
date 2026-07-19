---
name: Shelf-entry proposal
about: Propose a new off-the-shelf component on one of the ten extension points
labels: shelf
---

## The component

<!-- One sentence: what it is and when a modeller reaches for it. -->

## Extension point

<!-- Which of the ten points (the folders under src/extensions/) it lives
     on, and which trait it implements. If no point fits, read the
     "When nothing fits" section of the extending guide first; most ideas
     are a response-family augmentation in disguise, and the second answer
     is embedding. -->

## The maths

<!-- The distribution / update / distance in question, with a citation if
     it comes from a paper. Conjugate or augmentation-based families only
     on the deep seam (the scope rule in the extending guide). -->

## Validation plan

<!-- Which conformance check applies (`conformance::check_*` for the
     point), and, for anything touching a sampling kernel, the battery leg
     you would add (Geweke/SBC through `calibration::getting_it_right`).
     Adaptive components need their own MH correction and a battery run;
     see the inclusion point's exactness warning. -->
