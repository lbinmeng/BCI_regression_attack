#Target attack for regression in EEG-based BCIs

## attack.py
perform attack on DNN and Ridge, and plot the example of adversarial example.

## driving_attack.py
perform attack on all driving dataset, both within-subject and cross-subject.

## PVT_attack.py
perform attack on all pvt dataset, both within-subject and cross-suject.

## influence_of_parameters.py
validate the influence of the parameters
* the influence of the iteration in IFGSM
* the influence of the Epsilon in IFGSM
* the Influence of the constant c in C&W
* different target value

## transferability
validate the transferability of the adversarial example

## result.py
show the result of some experiments

## analyze_adversarial_example.py 
plot the spectrogram of the adversarial example
