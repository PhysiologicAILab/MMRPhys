conda remove --name mmrphys --all -y
conda create -n mmrphys python=3.11.2 pytorch=2.0.0 torchvision=0.15.1 torchaudio=2.0.1 cudatoolkit=11.7 -c pytorch -q -y
