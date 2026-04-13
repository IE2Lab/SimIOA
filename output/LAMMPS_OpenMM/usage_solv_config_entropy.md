python solvation_configurational_entropy.py \
  --topology config.pdb \
  --trajectory nvt3.lammpsdump \
  --solute "resname Li" \
  --shell-species FSI "resname fsa and (name O* F*)" 2.75 \
  --shell-species P3FO "resname p3f and (name O* F*)" 2.85 \
  --start-ps 500 \
  --sample-every 50 \
  --output-prefix P3FO_LiFSI
